//! Per-chat coding-sandbox VM lifecycle.
//!
//! `VmManager` is the sole consumer of [`EventType::CodingVmEnsureRequested`]
//! and [`EventType::CodingVmTeardownRequested`] on the `iron-events` bus.
//! It owns the `ready` map of live sandbox containers and translates
//! incoming requests into `systemctl start` / `systemctl stop` plus
//! `docker exec` orchestration. No consumer crate holds a direct
//! handle on it — chatorchestrator and webui interact entirely
//! through the bus.
//!
//! Cross-crate dep graph: `ironllm` (binary) → `vm_manager` →
//! `iron-events`. Identical to `toolexecutor` and `vm_enforcer`.
//!
//! Lifecycle boundaries:
//!
//! | Trigger                                       | Method                      |
//! |-----------------------------------------------|-----------------------------|
//! | [`EventType::CodingVmEnsureRequested`]        | `ensure_vm`                 |
//! | [`EventType::CodingVmTeardownRequested`] (`destroy=false`) | `teardown_vm`   |
//! | [`EventType::CodingVmTeardownRequested`] (`destroy=true`)  | `teardown_and_destroy` |
//! | `docker wait` catches an external exit (4h killer, OOM, manual `docker rm`) | watcher publishes `CodingVmTornDown` |
//! | Process boot                                  | `rehydrate_from_docker`     |
//!
//! ### `EventId` discipline
//!
//! Mirrors the `toolexecutor` pattern verbatim:
//!
//! * `CodingVmEnsureRequested` arrives with an `EventId` derived by the
//!   requester. We register the spawned handler in `in_flight_ensures`
//!   keyed by that id, run the boot, then publish `CodingVmEnsureResponded`
//!   via [`IronEvent::traceable_event`] echoing the same id back to
//!   `event.sender`. Identical shape to
//!   [`toolexecutor::ToolExecutor`]'s `ToolInvoke` → `ToolResponse` round-trip.
//! * `CodingVmTeardownRequested` carries an `EventId` for trace
//!   correlation. We propagate that id onto the resulting
//!   `CodingVmTornDown` envelope and include `event.sender` in the
//!   targets list so the originating publisher can evict its own
//!   pending-teardown entry (foundation for future dead-letter / replay).
//! * Unexpected container exits (watcher path) and rehydrate boots
//!   carry [`EventId::NONE`] — no originator.
//!
//! ### Concurrency
//!
//! `tokio::sync::Mutex<HashMap<_,_>>` throughout. The maps are touched
//! once per tool call and once per lifecycle event — adding `DashMap`
//! would be more deps for no measurable win.

use std::collections::HashMap;
use std::sync::Arc;

use iron_events::{
    CodingVmBootedPayload, CodingVmEnsureRequestedPayload, CodingVmEnsureRespondedPayload,
    CodingVmTeardownRequestedPayload, CodingVmTornDownPayload, EventBus, EventId, EventType,
    IronEvent, Subscriber,
};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

/// Host-side directory holding per-chat loopback images + bind-mount points.
pub const FOLDERS_ROOT: &str = "/mnt/docs/agentFolders";

/// Information we carry per live VM. Cloned out of the lock.
#[derive(Clone, Debug)]
pub struct VmState {
    pub chat_id: String,
    pub user_name: String,
    /// Name docker knows the container by (always `agent-<chat_id>`).
    pub container_name: String,
    /// `true` when this VM was just spun up from scratch by `do_boot`
    /// (manage-coding-session.sh wiped `/workspace/working/`). `false`
    /// for containers rehydrated from a pre-existing `docker ps` hit.
    /// The fast path in `ensure_vm_internal` overrides this to `false`
    /// on the returned clone — only callers whose request actually
    /// triggered a boot ever see `true`.
    pub fresh_boot: bool,
}

impl VmState {
    /// Systemd template instance name: `<user>__<chat_id>`.
    pub fn instance(&self) -> String {
        format!("{}__{}", self.user_name, self.chat_id)
    }
}

/// Compose the service unit name for `systemctl` args.
pub fn service_name(user_name: &str, chat_id: &str) -> String {
    format!("agent-vm@{user_name}__{chat_id}.service")
}

/// Docker container name for the chat (unique by `chat_id` alone).
pub fn container_name(chat_id: &str) -> String {
    format!("agent-{chat_id}")
}

/// One spawned ensure-handler task, registered in `in_flight_ensures`
/// at dispatch time and removed on completion. `chat_id`/`user_name`
/// travel alongside the `JoinHandle` so log lines fired at cancel
/// or timeout can carry full traceability without a second lookup.
///
/// `handle` is held so a future `CodingVmEnsureCanceled` listener arm
/// can `entry.handle.abort()` to drop the in-flight task at its next
/// `.await`, mirroring the `ToolCanceled` flow in `toolexecutor`.
/// Until that cancellation event exists the field is held but not
/// read — `dropping` the `JoinHandle` does *not* abort a tokio task,
/// so retaining it is the only way to keep the abort capability
/// available for the dead-letter / replay daemon to use later.
struct InFlightEnsure {
    #[allow(dead_code)]
    handle: JoinHandle<()>,
    chat_id: String,
    user_name: String,
}

/// One spawned teardown-handler task. Same shape and rationale as
/// `InFlightEnsure`; kept distinct so cancellation / sweep policies
/// can diverge later (a teardown sweep would be more aggressive
/// than an ensure sweep).
struct InFlightTeardown {
    #[allow(dead_code)]
    handle: JoinHandle<()>,
    chat_id: String,
    user_name: String,
}

/// Central per-orchestrator VM registry.
pub struct VmManager {
    /// VMs whose container is up and ready for `docker exec`.
    ready: Mutex<HashMap<String, VmState>>,
    /// VMs whose boot is in-flight. Waiters call `.notified().await`
    /// on the inner `Notify`; once boot finishes (either into `ready`
    /// or into a failure log), the entry is removed and waiters wake.
    booting: Mutex<HashMap<String, Arc<Notify>>>,
    /// `docker wait` watcher per chat. Fires when the container exits
    /// for any reason (4h `RuntimeMaxSec` killer, OOM, manual `docker rm`,
    /// daemon crash) and evicts the matching `ready` entry so the next
    /// `ensure_vm` re-boots instead of fast-pathing onto a dead name.
    /// Aborted on intentional teardown so only unintended exits reach
    /// the eviction code. Keyed by `chat_id`.
    watchers: Mutex<HashMap<String, JoinHandle<()>>>,
    /// In-flight ensure handlers keyed by the originating
    /// `CodingVmEnsureRequested` `EventId`. Inserted at dispatch,
    /// removed on completion. Mirrors `toolexecutor::ToolExecutor::in_flight`.
    in_flight_ensures: Arc<Mutex<HashMap<EventId, InFlightEnsure>>>,
    /// In-flight teardown handlers, same correlation discipline.
    in_flight_teardowns: Arc<Mutex<HashMap<EventId, InFlightTeardown>>>,
    /// Process-wide event bus. Used to publish VM-lifecycle events
    /// and to receive ensure/teardown requests.
    bus: EventBus,
}

impl VmManager {
    pub fn new(bus: EventBus) -> Arc<Self> {
        Arc::new(Self {
            ready: Mutex::new(HashMap::new()),
            booting: Mutex::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
            in_flight_ensures: Arc::new(Mutex::new(HashMap::new())),
            in_flight_teardowns: Arc::new(Mutex::new(HashMap::new())),
            bus,
        })
    }

    /// Subscribe on the bus and spawn the listener. MUST be called
    /// before any publisher of [`EventType::CodingVmEnsureRequested`]
    /// or [`EventType::CodingVmTeardownRequested`] — the bus has no
    /// replay, so events fired before this lands are silently dropped.
    pub async fn start_event_listener(self: &Arc<Self>) {
        let (_id, mut rx) = self.bus.subscribe(Subscriber::VmManager.as_str()).await;
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        let Some(state) = weak.upgrade() else { break };
                        match EventType::from_name(&event.event_type) {
                            Some(EventType::Ping) => {
                                tracing::info!(
                                    target: "iron_events",
                                    "vm_manager received Ping from {}",
                                    event.sender,
                                );
                            }
                            Some(EventType::CodingVmEnsureRequested) => {
                                state.dispatch_ensure(event).await;
                            }
                            Some(EventType::CodingVmTeardownRequested) => {
                                state.dispatch_teardown(event).await;
                            }
                            Some(_) => {
                                tracing::debug!(
                                    target: "iron_events",
                                    "vm_manager ignored event_type='{}' from {}",
                                    event.event_type,
                                    event.sender,
                                );
                            }
                            None => {
                                tracing::debug!(
                                    target: "iron_events",
                                    "vm_manager ignored unknown event_type='{}' from {}",
                                    event.event_type,
                                    event.sender,
                                );
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            target: "iron_events",
                            "vm_manager dropped {n} events (listener lagged)",
                        );
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    /// Spawn the rehydrate task at process start. Detaches the
    /// returned `JoinHandle`; caller drops it. Repopulates `ready`
    /// from live container labels so a mid-session `IronLLM` restart
    /// doesn't orphan running sandboxes.
    pub fn spawn_rehydrate(self: &Arc<Self>) -> JoinHandle<()> {
        let mgr = self.clone();
        tokio::spawn(async move {
            if let Err(e) = mgr.rehydrate_from_docker().await {
                tracing::warn!("VM rehydrate on startup failed: {e}");
            }
        })
    }

    // ── Listener dispatch helpers ────────────────────────────────────

    async fn dispatch_ensure(self: &Arc<Self>, event: IronEvent) {
        let event_id = event.id;
        let sender = event.sender.clone();
        let payload: CodingVmEnsureRequestedPayload =
            match serde_json::from_value(event.data.clone()) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        target: "iron_events",
                        "vm_manager got malformed CodingVmEnsureRequested from {}: {e}",
                        event.sender,
                    );
                    return;
                }
            };

        let chat_id = payload.chat_id.clone();
        let user_name = payload.user_name.clone();
        let mgr = self.clone();
        let in_flight = self.in_flight_ensures.clone();
        let id_for_dispatch = event_id;

        let handle = tokio::spawn(async move {
            let result = mgr.ensure_vm_internal(&chat_id, &user_name, id_for_dispatch).await;
            let resp = match result {
                Ok(vm) => CodingVmEnsureRespondedPayload {
                    container_name: vm.container_name,
                    fresh_boot: vm.fresh_boot,
                    error: None,
                },
                Err(e) => CodingVmEnsureRespondedPayload {
                    container_name: String::new(),
                    fresh_boot: false,
                    error: Some(e),
                },
            };
            let data =
                serde_json::to_value(&resp).expect("CodingVmEnsureRespondedPayload serializes");
            mgr.bus
                .publish(IronEvent::traceable_event(
                    id_for_dispatch,
                    Subscriber::VmManager.as_str(),
                    EventType::CodingVmEnsureResponded.as_str(),
                    vec![sender.clone()],
                    data,
                ))
                .await;
            // Self-deregister and log the close-out using the
            // per-entry chat_id/user_name so traceability survives
            // the spawn boundary without a second lookup.
            if let Some(entry) = in_flight.lock().await.remove(&id_for_dispatch) {
                tracing::debug!(
                    target: "iron_events",
                    chat_id = %entry.chat_id,
                    user_name = %entry.user_name,
                    event_id = %id_for_dispatch,
                    "vm_manager: ensure handler completed",
                );
            }
        });

        self.in_flight_ensures.lock().await.insert(
            event_id,
            InFlightEnsure {
                handle,
                chat_id: payload.chat_id,
                user_name: payload.user_name,
            },
        );
    }

    async fn dispatch_teardown(self: &Arc<Self>, event: IronEvent) {
        let event_id = event.id;
        let sender = event.sender.clone();
        let payload: CodingVmTeardownRequestedPayload =
            match serde_json::from_value(event.data.clone()) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(
                        target: "iron_events",
                        "vm_manager got malformed CodingVmTeardownRequested from {}: {e}",
                        event.sender,
                    );
                    return;
                }
            };

        let chat_id = payload.chat_id.clone();
        let user_name = payload.user_name.clone();
        let destroy = payload.destroy;
        let mgr = self.clone();
        let in_flight = self.in_flight_teardowns.clone();
        let id_for_dispatch = event_id;

        let handle = tokio::spawn(async move {
            let outcome = if destroy {
                mgr.teardown_and_destroy_internal(&chat_id, &user_name, id_for_dispatch, &sender)
                    .await
            } else {
                mgr.teardown_vm_internal(&chat_id, &user_name, id_for_dispatch, &sender)
                    .await
            };
            if let Err(e) = outcome {
                tracing::warn!(
                    target: "iron_events",
                    chat_id = %chat_id,
                    destroy,
                    "vm_manager teardown failed: {e}",
                );
            }
            if let Some(entry) = in_flight.lock().await.remove(&id_for_dispatch) {
                tracing::debug!(
                    target: "iron_events",
                    chat_id = %entry.chat_id,
                    user_name = %entry.user_name,
                    event_id = %id_for_dispatch,
                    destroy,
                    "vm_manager: teardown handler completed",
                );
            }
        });

        self.in_flight_teardowns.lock().await.insert(
            event_id,
            InFlightTeardown {
                handle,
                chat_id: payload.chat_id,
                user_name: payload.user_name,
            },
        );
    }

    // ── Public introspection (test + main only — not used cross-crate) ──

    /// Peek — is there a ready VM for this chat? Non-blocking beyond
    /// the lock acquisition.
    pub async fn peek(&self, chat_id: &str) -> Option<VmState> {
        self.ready.lock().await.get(chat_id).cloned()
    }

    // ── Boot / teardown internals (driven only by listener) ──────────

    /// Ensure a VM exists for this chat, returning its state once the
    /// container is up. Idempotent: concurrent callers coalesce onto
    /// a single in-flight boot; already-ready VMs return instantly.
    ///
    /// `originating_event_id` is propagated onto the resulting
    /// `CodingVmBooted` envelope when this call actually triggers a
    /// fresh boot, so the trace `EnsureRequested → EnsureResponded →
    /// CodingVmBooted` shares one id. `EventId::NONE` is fine and
    /// just means the boot leg of the trace is anonymous.
    async fn ensure_vm_internal(
        self: &Arc<Self>,
        chat_id: &str,
        user_name: &str,
        originating_event_id: EventId,
    ) -> Result<VmState, String> {
        // Fast path — VM is in the ready map. Verify the container is
        // actually alive before trusting the entry; if it crashed and
        // the watcher missed the exit, evict and fall through to boot.
        if let Some(vm) = self.peek(chat_id).await {
            if is_container_running(&vm.container_name).await {
                let mut vm = vm;
                vm.fresh_boot = false;
                return Ok(vm);
            }
            tracing::warn!(
                chat_id,
                container = %vm.container_name,
                "fast-path probe: container not running, evicting stale entry",
            );
            self.ready.lock().await.remove(chat_id);
            if let Some(h) = self.watchers.lock().await.remove(chat_id) {
                h.abort();
            }
        }

        // Claim the boot slot, or piggy-back on an in-flight one.
        let notify = {
            let mut booting = self.booting.lock().await;
            if let Some(existing) = booting.get(chat_id) { existing.clone() } else {
                let n = Arc::new(Notify::new());
                booting.insert(chat_id.to_string(), n.clone());

                let mgr = self.clone();
                let cid = chat_id.to_string();
                let uname = user_name.to_string();
                let n2 = n.clone();
                tokio::spawn(async move {
                    let outcome = do_boot(&cid, &uname).await;
                    match outcome {
                        Ok(actually_fresh) => {
                            mgr.ready.lock().await.insert(
                                cid.clone(),
                                VmState {
                                    chat_id: cid.clone(),
                                    user_name: uname.clone(),
                                    container_name: container_name(&cid),
                                    fresh_boot: actually_fresh,
                                },
                            );
                            // Arm the exit watcher. Abort any stale watcher
                            // first so we never have two running for the
                            // same chat_id.
                            {
                                let mut w = mgr.watchers.lock().await;
                                if let Some(h) = w.remove(&cid) {
                                    h.abort();
                                }
                                let handle = tokio::spawn(watch_container_exit(
                                    mgr.clone(),
                                    cid.clone(),
                                ));
                                w.insert(cid.clone(), handle);
                            }
                            // Announce the boot. `fresh` matches the
                            // `actually_fresh` flag so chatorchestrator's
                            // listener can decide whether to reset
                            // chat-scoped trackers (read-before-edit set,
                            // tool-call loop history) — those mirror
                            // /workspace/working/, which only gets wiped
                            // when the unit was inactive at start time.
                            //
                            // The originating EventId is propagated onto
                            // the boot envelope so the trace
                            // `EnsureRequested → EnsureResponded →
                            // CodingVmBooted` shares one id.
                            publish_booted(
                                &mgr.bus,
                                &cid,
                                &uname,
                                actually_fresh,
                                originating_event_id,
                            )
                            .await;
                        }
                        Err(e) => {
                            tracing::error!(chat_id = %cid, "VM boot failed: {e}");
                        }
                    }
                    mgr.booting.lock().await.remove(&cid);
                    n2.notify_waiters();
                });

                n
            }
        };

        notify.notified().await;

        self.peek(chat_id)
            .await
            .ok_or_else(|| format!("VM boot failed for chat {chat_id}"))
    }

    /// Stop one chat's VM (preserves `outputs/`). Idempotent.
    /// Called from the teardown listener; not exposed as cross-crate
    /// API. `originating_event_id` and `originating_sender` are
    /// propagated onto the resulting `CodingVmTornDown` so the
    /// publisher can evict its pending entry.
    async fn teardown_vm_internal(
        &self,
        chat_id: &str,
        user_name: &str,
        originating_event_id: EventId,
        originating_sender: &str,
    ) -> Result<(), String> {
        let svc = service_name(user_name, chat_id);
        tracing::info!(chat_id, user_name, "systemctl stop {}", svc);
        let out = Command::new("systemctl")
            .args(["stop", &svc])
            .output()
            .await
            .map_err(|e| format!("systemctl stop spawn: {e}"))?;
        self.ready.lock().await.remove(chat_id);
        if let Some(h) = self.watchers.lock().await.remove(chat_id) {
            h.abort();
        }
        publish_torn_down(
            &self.bus,
            chat_id,
            originating_event_id,
            Some(originating_sender),
        )
        .await;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !stderr.contains("not loaded") && !stderr.contains("not-found") {
                return Err(format!("systemctl stop: {}", stderr.trim()));
            }
        }
        Ok(())
    }

    /// Stop + wipe the on-disk chat image. Destroys `outputs/` as
    /// well — no recovery. Used by the chat-delete path.
    async fn teardown_and_destroy_internal(
        &self,
        chat_id: &str,
        user_name: &str,
        originating_event_id: EventId,
        originating_sender: &str,
    ) -> Result<(), String> {
        self.teardown_vm_internal(chat_id, user_name, originating_event_id, originating_sender)
            .await?;
        let dir = format!("{FOLDERS_ROOT}/{chat_id}");
        tracing::info!(chat_id, "rm -rf {}", dir);
        let out = Command::new("rm")
            .args(["-rf", "--", &dir])
            .output()
            .await
            .map_err(|e| format!("rm -rf spawn: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!("rm -rf {dir}: {}", stderr.trim()));
        }
        Ok(())
    }

    /// Walk `docker ps --filter label=coding_vm=true` and repopulate
    /// the ready map. Retries `docker ps` for ~5 s in case the daemon
    /// is still coming up.
    pub async fn rehydrate_from_docker(self: &Arc<Self>) -> Result<(), String> {
        let mut stdout_owned = String::new();
        for attempt in 0..10 {
            let out = Command::new("docker")
                .args([
                    "ps",
                    "--filter",
                    "label=coding_vm=true",
                    "--format",
                    "{{.Names}}\t{{.Label \"user_name\"}}\t{{.Label \"chat_id\"}}",
                ])
                .output()
                .await
                .map_err(|e| format!("docker ps spawn: {e}"))?;
            if out.status.success() {
                stdout_owned = String::from_utf8_lossy(&out.stdout).into_owned();
                break;
            }
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if attempt == 9 {
                tracing::warn!(
                    "rehydrate gave up after 10 attempts: docker ps failed: {stderr}"
                );
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        let mut ready = self.ready.lock().await;
        let mut watchers = self.watchers.lock().await;
        for line in stdout_owned.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                continue;
            }
            let (name, user_name, chat_id) = (parts[0], parts[1], parts[2]);
            if chat_id.is_empty() || user_name.is_empty() {
                continue;
            }
            tracing::info!(
                chat_id,
                user_name,
                container = name,
                "rehydrated coding VM from docker labels",
            );
            ready.insert(
                chat_id.to_string(),
                VmState {
                    chat_id: chat_id.to_string(),
                    user_name: user_name.to_string(),
                    container_name: name.to_string(),
                    // Rehydrate → same container IronLLM saw before its
                    // own restart. Filesystem is intact, no split brain,
                    // no reconcile message needed.
                    fresh_boot: false,
                },
            );
            let handle =
                tokio::spawn(watch_container_exit(self.clone(), chat_id.to_string()));
            watchers.insert(chat_id.to_string(), handle);
            // Rehydrate has no originating EventId. `fresh = false`
            // because the container survived → /working was not wiped.
            publish_booted(&self.bus, chat_id, user_name, false, EventId::NONE).await;
        }
        Ok(())
    }
}

async fn is_container_running(container: &str) -> bool {
    Command::new("docker")
        .args(["inspect", "-f", "{{.State.Running}}", container])
        .output()
        .await
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "true")
        .unwrap_or(false)
}

// ── Private: shell out to systemctl, wait for docker to report the
//    container is up. Returns `true` when this call actually booted
//    the VM fresh (systemd unit was inactive), `false` when the unit
//    was already active. ──────────────────────────────────────────────
async fn do_boot(chat_id: &str, user_name: &str) -> Result<bool, String> {
    let svc = service_name(user_name, chat_id);

    let was_active = Command::new("systemctl")
        .args(["is-active", "--quiet", &svc])
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);

    let ctr = container_name(chat_id);

    // If systemd says the unit is active, check whether the container
    // is actually alive. If it is, we can skip the boot entirely.
    // If the container is gone (manual docker rm, OOM) but the oneshot
    // unit lingers as "active (exited)", `systemctl start` is a no-op
    // — use `restart` to force re-execution of the session script.
    if was_active {
        if is_container_running(&ctr).await {
            tracing::info!(chat_id, container = %ctr, "coding VM already running (not in map)");
            return Ok(false);
        }
        tracing::warn!(
            chat_id,
            container = %ctr,
            "systemd unit active but container gone — restarting",
        );
    }

    let verb = if was_active { "restart" } else { "start" };

    tracing::info!(chat_id, user_name, was_active, "systemctl {} {}", verb, svc);

    let out = Command::new("systemctl")
        .args([verb, &svc])
        .output()
        .await
        .map_err(|e| format!("systemctl {verb} spawn: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("systemctl {verb}: {}", stderr.trim()));
    }

    for _ in 0..20 {
        let probe = Command::new("docker")
            .args(["inspect", "-f", "{{.State.Running}}", &ctr])
            .output()
            .await
            .map_err(|e| format!("docker inspect spawn: {e}"))?;
        if probe.status.success()
            && String::from_utf8_lossy(&probe.stdout).trim() == "true"
        {
            tracing::info!(chat_id, container = %ctr, was_active, "coding VM ready");
            // Both paths (fresh start and restart of stale unit) run
            // manage-coding-session.sh which wipes /workspace/working/.
            // Either way the model's files are gone → fresh boot.
            return Ok(true);
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    Err(format!("container {ctr} not in Running state after 5s"))
}

// ── Private: block on `docker wait <container>` and evict on exit.
//    Handles every non-systemd exit path. Intentional teardowns
//    abort this task, so only *unexpected* exits reach here. ──────
fn watch_container_exit(
    mgr: Arc<VmManager>,
    chat_id: String,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(watch_container_exit_inner(mgr, chat_id))
}

async fn watch_container_exit_inner(mgr: Arc<VmManager>, chat_id: String) {
    let ctr = container_name(&chat_id);
    let wait_ok = Command::new("docker")
        .args(["wait", &ctr])
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !wait_ok {
        // `docker wait` can fail if the container was removed before
        // the wait attached, or the daemon hiccupped. Fall back to a
        // probe: if the container is genuinely still running, re-arm
        // the watcher instead of wrongly evicting.
        if is_container_running(&ctr).await {
            tracing::warn!(
                chat_id,
                container = %ctr,
                "docker wait failed but container still running — re-arming watcher",
            );
            let handle = tokio::spawn(watch_container_exit(mgr.clone(), chat_id.clone()));
            mgr.watchers.lock().await.insert(chat_id, handle);
            return;
        }
        tracing::warn!(
            chat_id,
            container = %ctr,
            "docker wait failed and container is down — evicting",
        );
    }

    mgr.ready.lock().await.remove(&chat_id);
    mgr.watchers.lock().await.remove(&chat_id);
    tracing::info!(
        chat_id,
        container = %ctr,
        "container exited; evicted ready entry",
    );
    // Unexpected exit → no originating publisher to notify back.
    // EventId::NONE on the envelope; targets stay at the canonical
    // VM-lifecycle subscribers.
    publish_torn_down(&mgr.bus, &chat_id, EventId::NONE, None).await;
}

// ── Event publishing helpers ─────────────────────────────────────────

async fn publish_booted(
    bus: &EventBus,
    chat_id: &str,
    user_name: &str,
    fresh: bool,
    event_id: EventId,
) {
    let data = serde_json::to_value(CodingVmBootedPayload {
        chat_id: chat_id.to_string(),
        user_name: user_name.to_string(),
        fresh,
    })
    .expect("CodingVmBootedPayload always serializes");
    let envelope = if event_id == EventId::NONE {
        IronEvent::new(
            Subscriber::VmManager.as_str(),
            EventType::CodingVmBooted.as_str(),
            vec![
                Subscriber::VmEnforcer.as_str().to_string(),
                Subscriber::Chatorchestrator.as_str().to_string(),
            ],
            data,
        )
    } else {
        IronEvent::traceable_event(
            event_id,
            Subscriber::VmManager.as_str(),
            EventType::CodingVmBooted.as_str(),
            vec![
                Subscriber::VmEnforcer.as_str().to_string(),
                Subscriber::Chatorchestrator.as_str().to_string(),
            ],
            data,
        )
    };
    bus.publish(envelope).await;
}

async fn publish_torn_down(
    bus: &EventBus,
    chat_id: &str,
    event_id: EventId,
    originating_sender: Option<&str>,
) {
    let data = serde_json::to_value(CodingVmTornDownPayload {
        chat_id: chat_id.to_string(),
    })
    .expect("CodingVmTornDownPayload always serializes");
    let mut targets = vec![
        Subscriber::VmEnforcer.as_str().to_string(),
        Subscriber::Chatorchestrator.as_str().to_string(),
    ];
    // Echo back to the originating publisher so it can evict its
    // pending-teardown entry. Skip if the originator is already in
    // the canonical set (avoid duplicate delivery on the same target).
    if let Some(sender) = originating_sender {
        if !targets.iter().any(|t| t == sender) {
            targets.push(sender.to_string());
        }
    }
    let envelope = if event_id == EventId::NONE {
        IronEvent::new(
            Subscriber::VmManager.as_str(),
            EventType::CodingVmTornDown.as_str(),
            targets,
            data,
        )
    } else {
        IronEvent::traceable_event(
            event_id,
            Subscriber::VmManager.as_str(),
            EventType::CodingVmTornDown.as_str(),
            targets,
            data,
        )
    };
    bus.publish(envelope).await;
}
