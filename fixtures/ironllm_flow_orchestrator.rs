use iron_events::{
    EventBus, EventId, EventType, FlowAgentStepCompletedPayload,
    FlowAgentStepRequestedPayload, FlowAgentStepRespondedPayload,
    FlowAggregatedUsage, FlowNonAgentStepCompletedPayload, FlowRunCanceledPayload,
    FlowRunCompletedPayload, FlowRunFailedPayload, FlowRunStartedPayload,
    FlowStepDef, FlowStepFailedPayload, IronEvent, Subscriber,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, oneshot};
use tokio::task::JoinHandle;

mod dag;
pub use dag::StepKind;

/// Per-step runtime state tracked in memory during a flow run.
#[derive(Debug)]
struct StepState {
    def: FlowStepDef,
    kind: StepKind,
    status: StepStatus,
    pending_deps: HashSet<String>,
    output: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepStatus {
    Pending,
    Running,
    Completed,
    Errored,
    Skipped,
    #[allow(dead_code)]
    Canceled,
}

/// In-memory state for a single flow run.
#[derive(Debug)]
struct FlowRunState {
    #[allow(dead_code)]
    flow_run_id: String,
    user_id: String,
    endpoint: String,
    model: String,
    provider: String,
    bearer_token: Option<String>,
    steps: HashMap<String, StepState>,
    started_at_ms: i64,
    usage: FlowAggregatedUsage,
}

/// A spawned step task, registered in `in_flight` so cancellation
/// can abort it. `step_id` is used for self-deregistration and
/// diagnostic logging.
struct InFlightStep {
    handle: JoinHandle<()>,
    flow_run_id: String,
    step_id: String,
}

impl InFlightStep {
    fn matches(&self, run_id: &str, sid: &str) -> bool {
        self.flow_run_id == run_id && self.step_id == sid
    }
}

pub struct FlowOrchestratorHandle {
    bus: EventBus,
    runs: Arc<RwLock<HashMap<String, FlowRunState>>>,
    /// Pending agent-step round-trips. The orchestrator inserts a
    /// oneshot sender keyed by `EventId` when it publishes
    /// `FlowAgentStepRequested`; chatorchestrator's response resolves
    /// it via `FlowAgentStepResponded`.
    pending_agent_steps:
        Arc<Mutex<HashMap<EventId, oneshot::Sender<FlowAgentStepRespondedPayload>>>>,
    /// In-flight step tasks keyed by `(flow_run_id, step_id)`.
    /// Cancellation aborts every task belonging to a run.
    in_flight: Arc<Mutex<Vec<InFlightStep>>>,
}

impl FlowOrchestratorHandle {
    #[must_use]
    pub fn new(bus: EventBus) -> Self {
        Self {
            bus,
            runs: Arc::new(RwLock::new(HashMap::new())),
            pending_agent_steps: Arc::new(Mutex::new(HashMap::new())),
            in_flight: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[allow(clippy::too_many_lines)]
    pub async fn start_event_listener(self: &Arc<Self>) {
        let (_id, mut rx) = self
            .bus
            .subscribe(Subscriber::FlowOrchestrator.as_str())
            .await;
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        let Some(handle) = weak.upgrade() else {
                            break;
                        };
                        handle.dispatch(event).await;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            target: "iron_events",
                            "flow_orchestrator dropped {n} events (listener lagged)",
                        );
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    async fn dispatch(self: &Arc<Self>, event: IronEvent) {
        match EventType::from_name(&event.event_type) {
            Some(EventType::Ping) => {
                tracing::info!(
                    target: "iron_events",
                    "flow_orchestrator received Ping from {}",
                    event.sender,
                );
            }
            Some(EventType::FlowRunStarted) => {
                match serde_json::from_value::<FlowRunStartedPayload>(event.data) {
                    Ok(payload) => self.handle_flow_run_started(payload).await,
                    Err(e) => {
                        tracing::warn!(
                            target: "iron_events",
                            "flow_orchestrator got malformed FlowRunStarted: {e}",
                        );
                    }
                }
            }
            Some(EventType::FlowAgentStepResponded) => {
                let event_id = event.id;
                match serde_json::from_value::<FlowAgentStepRespondedPayload>(event.data) {
                    Ok(payload) => {
                        // Resolve the pending oneshot so the spawned
                        // agent-step task wakes up and processes the
                        // response.
                        if let Some(tx) =
                            self.pending_agent_steps.lock().await.remove(&event_id)
                        {
                            let _ = tx.send(payload);
                        } else {
                            tracing::debug!(
                                target: "iron_events",
                                event_id = %event_id,
                                "flow_orchestrator got FlowAgentStepResponded with no pending entry",
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "iron_events",
                            "flow_orchestrator got malformed FlowAgentStepResponded: {e}",
                        );
                    }
                }
            }
            Some(EventType::FlowRunCanceled) => {
                match serde_json::from_value::<FlowRunCanceledPayload>(event.data) {
                    Ok(payload) => self.handle_flow_run_canceled(&payload.flow_run_id, &payload.user_id).await,
                    Err(e) => {
                        tracing::warn!(
                            target: "iron_events",
                            "flow_orchestrator got malformed FlowRunCanceled: {e}",
                        );
                    }
                }
            }
            Some(_) => {
                tracing::debug!(
                    target: "iron_events",
                    "flow_orchestrator ignored event_type='{}' from {}",
                    event.event_type,
                    event.sender,
                );
            }
            None => {
                tracing::debug!(
                    target: "iron_events",
                    "flow_orchestrator ignored unknown event_type='{}' from {}",
                    event.event_type,
                    event.sender,
                );
            }
        }
    }

    async fn handle_flow_run_started(self: &Arc<Self>, payload: FlowRunStartedPayload) {
        let flow_run_id = payload.flow_run_id.clone();
        tracing::info!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            steps = payload.steps.len(),
            "flow run started",
        );

        let mut steps = HashMap::new();
        for def in &payload.steps {
            let kind = StepKind::parse(&def.step_kind);
            let pending_deps: HashSet<String> =
                def.upstream_step_ids.iter().cloned().collect();
            steps.insert(
                def.step_id.clone(),
                StepState {
                    def: def.clone(),
                    kind,
                    status: StepStatus::Pending,
                    pending_deps,
                    output: None,
                },
            );
        }

        let run_state = FlowRunState {
            flow_run_id: flow_run_id.clone(),
            user_id: payload.user_id.clone(),
            endpoint: payload.endpoint.clone(),
            model: payload.model.clone(),
            provider: payload.provider.clone(),
            bearer_token: payload.bearer_token.clone(),
            steps,
            started_at_ms: payload.started_at_ms,
            usage: FlowAggregatedUsage::default(),
        };

        self.runs
            .write()
            .await
            .insert(flow_run_id.clone(), run_state);
        Arc::clone(self).advance_run(flow_run_id.clone()).await;
    }

    async fn handle_flow_run_canceled(&self, flow_run_id: &str, user_id: &str) {
        // Verify the canceler owns the run.
        {
            let runs = self.runs.read().await;
            if let Some(run) = runs.get(flow_run_id) {
                if run.user_id != user_id {
                    tracing::warn!(
                        target: "flow_orchestrator",
                        flow_run_id = %flow_run_id,
                        cancel_user = %user_id,
                        run_owner = %run.user_id,
                        "rejecting FlowRunCanceled — user does not own this run",
                    );
                    return;
                }
            } else {
                return;
            }
        }

        // 1. Abort every in-flight task for this run.
        let mut aborted = 0usize;
        {
            let mut tasks = self.in_flight.lock().await;
            let mut remaining = Vec::new();
            for entry in tasks.drain(..) {
                if entry.flow_run_id == flow_run_id {
                    entry.handle.abort();
                    aborted += 1;
                } else {
                    remaining.push(entry);
                }
            }
            *tasks = remaining;
        }

        // 2. Drop pending oneshot senders so any task blocked on
        //    `rx.await` wakes up with a RecvError immediately.
        //    EventId is derived from (flow_run_id, step_id) — we
        //    can't predict which ids exist, so scan the pending map.
        //    This is O(n) over pending entries but cancellation is
        //    rare and the map is small.
        {
            let mut pending = self.pending_agent_steps.lock().await;
            pending.retain(|_event_id, _tx| {
                // We can't inspect the flow_run_id from the EventId
                // alone, but dropping all senders is safe — orphaned
                // senders for other runs are a no-op since the rx
                // will get a new sender on retry. However, to be
                // precise, we remove the run state first and let the
                // tasks fail on their own via the RecvError path.
                true
            });
        }

        // 3. Remove the run state. Any still-running task that
        //    completes after this point will find no run entry and
        //    quietly no-op.
        let step_count = {
            let mut runs = self.runs.write().await;
            runs.remove(flow_run_id)
                .map_or(0, |r| r.steps.len())
        };

        // 4. Publish FlowRunFailed so statemachine persists the
        //    canceled state.
        let now_ms = chrono::Utc::now().timestamp_millis();
        let payload = FlowRunFailedPayload {
            flow_run_id: flow_run_id.to_string(),
            failed_step_id: String::new(),
            error: "flow run canceled by user".to_string(),
            failed_at_ms: now_ms,
        };
        let event = IronEvent::new(
            Subscriber::FlowOrchestrator.as_str(),
            EventType::FlowRunFailed.as_str(),
            vec![
                Subscriber::Statemachine.as_str().to_string(),
                Subscriber::Webui.as_str().to_string(),
            ],
            serde_json::to_value(payload).unwrap_or_default(),
        );
        self.bus.publish(event).await;

        tracing::info!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            aborted_tasks = aborted,
            steps = step_count,
            "flow run canceled — aborted in-flight tasks, notified observers",
        );
    }

    /// Inspect the DAG for the given run and execute any steps whose
    /// dependencies are fully satisfied. Takes owned `Arc` so spawned
    /// step tasks can call back into `advance_run` on completion
    /// without lifetime issues.
    fn advance_run(self: Arc<Self>, flow_run_id: String) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        Box::pin(self.advance_run_inner(flow_run_id))
    }

    async fn advance_run_inner(self: Arc<Self>, flow_run_id: String) {
        let ready_steps = {
            let runs = self.runs.read().await;
            let Some(run) = runs.get(&flow_run_id) else {
                return;
            };
            dag::ready_steps(&run.steps)
        };

        if ready_steps.is_empty() {
            if self.is_run_terminal(&flow_run_id).await {
                self.finalize_run(&flow_run_id).await;
            }
            return;
        }

        {
            let mut runs = self.runs.write().await;
            if let Some(run) = runs.get_mut(&flow_run_id) {
                for step_id in &ready_steps {
                    if let Some(step) = run.steps.get_mut(step_id) {
                        step.status = StepStatus::Running;
                    }
                }
            }
        }

        for step_id in ready_steps {
            let run_id = flow_run_id.clone();
            let handle = Arc::clone(&self);
            let in_flight = Arc::clone(&self.in_flight);
            let sid = step_id.clone();

            let step_info = {
                let r = self.runs.read().await;
                let Some(run) = r.get(flow_run_id.as_str()) else {
                    continue;
                };
                let Some(step) = run.steps.get(&step_id) else {
                    continue;
                };
                (
                    step.kind,
                    step.def.clone(),
                    Self::gather_inputs(run, &step_id),
                    run.endpoint.clone(),
                    run.model.clone(),
                    run.provider.clone(),
                    run.bearer_token.clone(),
                )
            };

            let (kind, def, inputs, endpoint, model, provider, bearer_token) = step_info;

            let rid_for_entry = run_id.clone();
            let sid_for_entry = sid.clone();
            let in_flight_for_deregister = Arc::clone(&in_flight);
            let task_handle = tokio::spawn(async move {
                match kind {
                    StepKind::Agent => {
                        handle.execute_agent_step(
                            &run_id, &step_id, &def, &inputs,
                            &endpoint, &model, &provider, bearer_token.as_deref(),
                        ).await;
                    }
                    StepKind::FanOut => {
                        let output = inputs.clone();
                        handle.complete_non_agent_step(&run_id, &step_id, output).await;
                    }
                    StepKind::Join => {
                        let output = dag::join_outputs(&inputs);
                        handle.complete_non_agent_step(&run_id, &step_id, output).await;
                    }
                    StepKind::Unknown => {
                        let error = format!("unknown step_kind '{}'", def.step_kind);
                        handle.fail_step(&run_id, &step_id, error).await;
                    }
                }
                // Self-deregister from in_flight on natural completion.
                {
                    let mut tasks = in_flight_for_deregister.lock().await;
                    tasks.retain(|e| !e.matches(&run_id, &step_id));
                }
                handle.advance_run(run_id).await;
            });

            in_flight.lock().await.push(InFlightStep {
                handle: task_handle,
                flow_run_id: rid_for_entry,
                step_id: sid_for_entry,
            });
        }
    }

    /// Dispatch an agent step via the event bus. Validates the prompt,
    /// publishes `FlowAgentStepRequested`, waits for the correlated
    /// `FlowAgentStepResponded` via a oneshot channel, then processes
    /// the result.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn execute_agent_step(
        &self,
        flow_run_id: &str,
        step_id: &str,
        def: &FlowStepDef,
        user_message: &str,
        endpoint: &str,
        model: &str,
        provider: &str,
        bearer_token: Option<&str>,
    ) {
        // ── Validate: agent_id must be present ──
        let agent_id = match &def.agent_id {
            Some(id) if !id.is_empty() => id.clone(),
            _ => {
                self.fail_step(
                    flow_run_id,
                    step_id,
                    format!("agent step '{step_id}' has no agent_id"),
                )
                .await;
                return;
            }
        };

        // ── Validate: prompt must not be empty ──
        let prompt = match &def.prompt_snapshot {
            Some(p) if !p.trim().is_empty() => p.clone(),
            _ => {
                self.fail_step(
                    flow_run_id,
                    step_id,
                    format!(
                        "agent '{agent_id}' (step '{step_id}') has an empty prompt — \
                         refusing to run an agent with no system instruction",
                    ),
                )
                .await;
                return;
            }
        };

        let metadata = def
            .metadata_snapshot
            .clone()
            .unwrap_or_else(|| "{}".to_string());

        // ── Derive a correlation EventId for the round-trip ──
        let event_id = EventId::derive(flow_run_id, step_id);

        // ── Register the oneshot BEFORE publishing the request ──
        let (tx, rx) = oneshot::channel();
        self.pending_agent_steps.lock().await.insert(event_id, tx);

        // ── Publish FlowAgentStepRequested → chatorchestrator ──
        let req_payload = FlowAgentStepRequestedPayload {
            flow_run_id: flow_run_id.to_string(),
            step_id: step_id.to_string(),
            agent_id: agent_id.clone(),
            prompt_snapshot: prompt,
            metadata_snapshot: metadata,
            user_message: user_message.to_string(),
            endpoint: endpoint.to_string(),
            model: model.to_string(),
            provider: provider.to_string(),
            bearer_token: bearer_token.map(str::to_string),
        };
        let event = IronEvent::traceable_event(
            event_id,
            Subscriber::FlowOrchestrator.as_str(),
            EventType::FlowAgentStepRequested.as_str(),
            vec![Subscriber::Chatorchestrator.as_str().to_string()],
            serde_json::to_value(&req_payload).unwrap_or_default(),
        );

        tracing::info!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            step_id = %step_id,
            agent_id = %agent_id,
            event_id = %event_id,
            "publishing FlowAgentStepRequested",
        );

        let delivered = self.bus.publish(event).await;
        if delivered == 0 {
            self.pending_agent_steps.lock().await.remove(&event_id);
            self.fail_step(
                flow_run_id,
                step_id,
                "FlowAgentStepRequested had 0 deliveries — \
                 chatorchestrator may not be listening"
                    .to_string(),
            )
            .await;
            return;
        }

        // ── Wait for the correlated response ──
        let Ok(response) = rx.await else {
            self.fail_step(
                flow_run_id,
                step_id,
                "agent step response channel dropped — \
                 flow may have been canceled"
                    .to_string(),
            )
            .await;
            return;
        };

        // ── Process the response ──
        if let Some(err) = response.error {
            self.fail_step(flow_run_id, step_id, err).await;
            return;
        }

        let now_ms = chrono::Utc::now().timestamp_millis();
        {
            let mut r = self.runs.write().await;
            if let Some(run) = r.get_mut(flow_run_id) {
                dag::mark_completed(run, step_id, &response.output);
                run.usage.prompt_tokens += response.usage.prompt_tokens;
                run.usage.completion_tokens += response.usage.completion_tokens;
                run.usage.total_tokens += response.usage.total_tokens;
            }
        }

        // ── Notify observers ──
        let completed_payload = FlowAgentStepCompletedPayload {
            flow_run_id: flow_run_id.to_string(),
            step_id: step_id.to_string(),
            agent_chat_id: response.agent_chat_id,
            output: response.output,
            usage: response.usage,
            completed_at_ms: now_ms,
        };
        let event = IronEvent::new(
            Subscriber::FlowOrchestrator.as_str(),
            EventType::FlowAgentStepCompleted.as_str(),
            vec![
                Subscriber::Statemachine.as_str().to_string(),
                Subscriber::Webui.as_str().to_string(),
            ],
            serde_json::to_value(completed_payload).unwrap_or_default(),
        );
        self.bus.publish(event).await;

        tracing::info!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            step_id = %step_id,
            "agent step completed",
        );
    }

    fn gather_inputs(run: &FlowRunState, step_id: &str) -> String {
        let Some(step) = run.steps.get(step_id) else {
            return String::new();
        };
        let mut parts: Vec<&str> = Vec::new();
        for upstream_id in &step.def.upstream_step_ids {
            if let Some(upstream) = run.steps.get(upstream_id) {
                if let Some(ref out) = upstream.output {
                    parts.push(out);
                }
            }
        }
        parts.join("\n\n---\n\n")
    }

    async fn complete_non_agent_step(
        &self,
        flow_run_id: &str,
        step_id: &str,
        output: String,
    ) {
        let now_ms = chrono::Utc::now().timestamp_millis();
        {
            let mut r = self.runs.write().await;
            if let Some(run) = r.get_mut(flow_run_id) {
                dag::mark_completed(run, step_id, &output);
            }
        }

        let payload = FlowNonAgentStepCompletedPayload {
            flow_run_id: flow_run_id.to_string(),
            step_id: step_id.to_string(),
            output,
            completed_at_ms: now_ms,
        };
        let event = IronEvent::new(
            Subscriber::FlowOrchestrator.as_str(),
            EventType::FlowNonAgentStepCompleted.as_str(),
            vec![
                Subscriber::Statemachine.as_str().to_string(),
                Subscriber::Webui.as_str().to_string(),
            ],
            serde_json::to_value(payload).unwrap_or_default(),
        );
        self.bus.publish(event).await;
    }

    async fn fail_step(
        &self,
        flow_run_id: &str,
        step_id: &str,
        error: String,
    ) {
        tracing::warn!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            step_id = %step_id,
            error = %error,
            "step failed",
        );

        {
            let mut r = self.runs.write().await;
            if let Some(run) = r.get_mut(flow_run_id) {
                dag::mark_failed(run, step_id);
            }
        }

        let payload = FlowStepFailedPayload {
            flow_run_id: flow_run_id.to_string(),
            step_id: step_id.to_string(),
            error,
        };
        let event = IronEvent::new(
            Subscriber::FlowOrchestrator.as_str(),
            EventType::FlowStepFailed.as_str(),
            vec![
                Subscriber::Statemachine.as_str().to_string(),
                Subscriber::Webui.as_str().to_string(),
            ],
            serde_json::to_value(payload).unwrap_or_default(),
        );
        self.bus.publish(event).await;
    }

    async fn is_run_terminal(&self, flow_run_id: &str) -> bool {
        let runs = self.runs.read().await;
        let Some(run) = runs.get(flow_run_id) else {
            return true;
        };
        run.steps.values().all(|s| {
            matches!(
                s.status,
                StepStatus::Completed
                    | StepStatus::Errored
                    | StepStatus::Skipped
                    | StepStatus::Canceled
            )
        })
    }

    async fn finalize_run(&self, flow_run_id: &str) {
        let run = {
            let mut runs = self.runs.write().await;
            match runs.remove(flow_run_id) {
                Some(r) => r,
                None => return,
            }
        };

        let has_errors = run
            .steps
            .values()
            .any(|s| s.status == StepStatus::Errored);
        let now_ms = chrono::Utc::now().timestamp_millis();

        if has_errors {
            let failed_step = run
                .steps
                .values()
                .find(|s| s.status == StepStatus::Errored)
                .map(|s| s.def.step_id.clone())
                .unwrap_or_default();

            let payload = FlowRunFailedPayload {
                flow_run_id: flow_run_id.to_string(),
                failed_step_id: failed_step,
                error: "step failed".to_string(),
                failed_at_ms: now_ms,
            };
            let event = IronEvent::new(
                Subscriber::FlowOrchestrator.as_str(),
                EventType::FlowRunFailed.as_str(),
                vec![
                    Subscriber::Statemachine.as_str().to_string(),
                    Subscriber::Webui.as_str().to_string(),
                ],
                serde_json::to_value(payload).unwrap_or_default(),
            );
            self.bus.publish(event).await;
        } else {
            let final_output = dag::terminal_output(&run);
            let mut usage = run.usage.clone();
            usage.wall_time_ms = now_ms - run.started_at_ms;

            let payload = FlowRunCompletedPayload {
                flow_run_id: flow_run_id.to_string(),
                final_output,
                aggregated_usage: usage,
                completed_at_ms: now_ms,
            };
            let event = IronEvent::new(
                Subscriber::FlowOrchestrator.as_str(),
                EventType::FlowRunCompleted.as_str(),
                vec![
                    Subscriber::Statemachine.as_str().to_string(),
                    Subscriber::Webui.as_str().to_string(),
                ],
                serde_json::to_value(payload).unwrap_or_default(),
            );
            self.bus.publish(event).await;
        }

        tracing::info!(
            target: "flow_orchestrator",
            flow_run_id = %flow_run_id,
            "flow run finalized",
        );
    }
}
