//! Runtime config — deserializes `ironllm.toml` into typed structs.
//!
//! After parsing, all values are also pushed into process env vars so
//! crates that still read `from_env()` (authorization, externalapi,
//! security headers) keep working with zero changes. Over time those
//! crates can migrate to accept the parsed struct directly.
//!
//! Policy:
//!   * **One file, one format.** `ironllm.toml` holds everything.
//!   * **Defaults fill gaps.** Any field not present uses `#[serde(default)]`.

use serde::Deserialize;
use std::path::Path;

// ── Defaults ────────────────────────────────────────────────────────

pub const DEFAULT_BIND_IP: &str = "0.0.0.0";

pub const DEFAULT_DASHBOARD_PORT: u16 = 16942;
pub const DEFAULT_DASHBOARD_API_BASE: &str = "/dash/v1";
pub const DEFAULT_DASHBOARD_DB_PATH: &str = "./data/dashboard/dashboard.db";

pub const DEFAULT_WEBCHAT_PORT: u16 = 14269;
pub const DEFAULT_WEBCHAT_API_BASE: &str = "/chat/v1";
pub const DEFAULT_WEBCHAT_DB_PATH: &str = "./data/webchat/webchat.db";

pub const DEFAULT_ADMIN_PORT: u16 = 14962;
pub const DEFAULT_ADMIN_API_BASE: &str = "/admin/v1";
pub const DEFAULT_ADMIN_DB_PATH: &str = "./data/admin/admin.db";

pub const DEFAULT_PASSTHROUGH_PORT: u16 = 16249;
pub const DEFAULT_PASSTHROUGH_API_BASE: &str = "/api/v1";
pub const DEFAULT_PASSTHROUGH_DB_PATH: &str = "./data/passthrough/passthrough.db";

pub const DEFAULT_LLM_TIMEOUT: u64 = 300;

// ── TOML schema ─────────────────────────────────────────────────────

/// Root config — mirrors the `ironllm.toml` file structure.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub application: ApplicationConfig,
    pub tls: TlsConfig,
    pub dashboard: ListenerConfig,
    pub webchat: ListenerConfig,
    pub admin: AdminListenerConfig,
    pub passthrough: PassthroughListenerConfig,
    pub llm: LlmConfig,
    pub tools: ToolsConfig,
    pub auth: AuthConfig,
    pub security: SecurityConfig,
    pub external_api: ExternalApiConfig,
    pub resources: ResourcesConfig,
    pub uploads: UploadsConfig,
    pub toolexecutor: ToolExecutorConfig,
    pub rag_search: RagSearchConfig,
    pub rag_ingestion: RagIngestionConfig,
    /// Optional test-user seed. If set and populated with a non-placeholder
    /// password, `IronLLM` boot upserts a `testuser` row via `seed_user`. The
    /// real value is expected to live in `ironllm.local.toml` (gitignored);
    /// the main `ironllm.toml` carries only a placeholder.
    pub test_seed: Option<TestSeedConfig>,
    /// Runtime crash-safety scratch space. Holds the embedded-PG `postgres`
    /// role password between an unclean shutdown and the next boot. Written
    /// only into `ironllm.local.toml` (never the main file), and removed
    /// again by the graceful-shutdown path in `cleanup.rs`. If a value is
    /// present at boot it means the previous run did not clean up — PG is
    /// still in scram mode and we must reuse this password instead of
    /// rotating, otherwise we lock ourselves out.
    pub runtime: Option<RuntimeConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            application: ApplicationConfig::default(),
            tls: TlsConfig::default(),
            dashboard: ListenerConfig {
                bind_ip: None,
                port: DEFAULT_DASHBOARD_PORT,
                api_base: DEFAULT_DASHBOARD_API_BASE.into(),
                db_path: DEFAULT_DASHBOARD_DB_PATH.into(),
            },
            webchat: ListenerConfig {
                bind_ip: None,
                port: DEFAULT_WEBCHAT_PORT,
                api_base: DEFAULT_WEBCHAT_API_BASE.into(),
                db_path: DEFAULT_WEBCHAT_DB_PATH.into(),
            },
            admin: AdminListenerConfig::default(),
            passthrough: PassthroughListenerConfig::default(),
            llm: LlmConfig::default(),
            tools: ToolsConfig::default(),
            auth: AuthConfig::default(),
            security: SecurityConfig::default(),
            external_api: ExternalApiConfig::default(),
            resources: ResourcesConfig::default(),
            uploads: UploadsConfig::default(),
            toolexecutor: ToolExecutorConfig::default(),
            rag_search: RagSearchConfig::default(),
            rag_ingestion: RagIngestionConfig::default(),
            test_seed: None,
            runtime: None,
        }
    }
}

// ── Runtime scratch (crash-safety for PG password) ─────────────────

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Embedded-PG `postgres` role password. Written to
    /// `ironllm.local.toml` immediately after generation so that an
    /// unclean shutdown (panic, SIGKILL, power loss) is recoverable on
    /// the next boot. Removed by the graceful-shutdown path.
    pub pg_role_password: Option<String>,
}

// ── Test seed (optional, for e2e integration tests) ─────────────────
//
// A placeholder value starting with `<` or shorter than 16 chars makes
// the auto-seed a no-op — see main.rs `run_server` for the skip logic.

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TestSeedConfig {
    pub password: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ApplicationConfig {
    pub bind_ip: String,
    pub claviger_dump: bool,
}

impl Default for ApplicationConfig {
    fn default() -> Self {
        Self {
            bind_ip: DEFAULT_BIND_IP.into(),
            claviger_dump: true,
        }
    }
}

// ── TLS ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct TlsConfig {
    /// Enable TLS on all listeners. When false, listeners serve plain HTTP.
    pub enabled: bool,
    /// Path to PEM-encoded certificate chain (fullchain.pem).
    pub cert_path: String,
    /// Path to PEM-encoded private key (privkey.pem).
    pub key_path: String,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cert_path: "./certs/fullchain.pem".into(),
            key_path: "./certs/privkey.pem".into(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ListenerConfig {
    /// Per-listener bind IP override. If set, overrides [application].`bind_ip`.
    pub bind_ip: Option<String>,
    pub port: u16,
    pub api_base: String,
    pub db_path: String,
}

impl ListenerConfig {
    pub fn bind_addr(&self, global_bind_ip: &str) -> String {
        let ip = self.bind_ip.as_deref().unwrap_or(global_bind_ip);
        format!("{}:{}", ip, self.port)
    }
}

// Separate types for admin/passthrough so Default works per-section.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AdminListenerConfig {
    pub bind_ip: Option<String>,
    pub port: u16,
    pub api_base: String,
    pub db_path: String,
}

impl Default for AdminListenerConfig {
    fn default() -> Self {
        Self {
            bind_ip: None,
            port: DEFAULT_ADMIN_PORT,
            api_base: DEFAULT_ADMIN_API_BASE.into(),
            db_path: DEFAULT_ADMIN_DB_PATH.into(),
        }
    }
}

impl AdminListenerConfig {
    pub fn bind_addr(&self, global_bind_ip: &str) -> String {
        let ip = self.bind_ip.as_deref().unwrap_or(global_bind_ip);
        format!("{}:{}", ip, self.port)
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PassthroughListenerConfig {
    pub bind_ip: Option<String>,
    pub port: u16,
    pub api_base: String,
    pub db_path: String,
}

impl Default for PassthroughListenerConfig {
    fn default() -> Self {
        Self {
            bind_ip: None,
            port: DEFAULT_PASSTHROUGH_PORT,
            api_base: DEFAULT_PASSTHROUGH_API_BASE.into(),
            db_path: DEFAULT_PASSTHROUGH_DB_PATH.into(),
        }
    }
}

impl PassthroughListenerConfig {
    pub fn bind_addr(&self, global_bind_ip: &str) -> String {
        let ip = self.bind_ip.as_deref().unwrap_or(global_bind_ip);
        format!("{}:{}", ip, self.port)
    }
}

// ── LLM servers ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub servers: Vec<LlmServer>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            servers: vec![LlmServer {
                name: "default".into(),
                address: "127.0.0.1".into(),
                port: 8078,
                timeout: DEFAULT_LLM_TIMEOUT,
            }],
        }
    }
}

/// A single LLM backend server.
#[derive(Debug, Clone, Deserialize)]
pub struct LlmServer {
    /// Human-readable name (shown in logs).
    #[serde(default = "default_server_name")]
    pub name: String,
    /// IP address or hostname.
    pub address: String,
    /// Port number.
    pub port: u16,
    /// HTTP stream timeout in seconds (covers full request lifetime).
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_server_name() -> String { "unnamed".into() }
fn default_timeout() -> u64 { DEFAULT_LLM_TIMEOUT }

impl LlmServer {
    /// Returns "address:port" string used by modelmonitor and orchestrator.
    pub fn host_port(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

impl std::fmt::Display for LlmServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{} \"{}\" ({}s timeout)", self.address, self.port, self.name, self.timeout)
    }
}

// ── Tool servers ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ToolsConfig {
    pub servers: Vec<ToolServer>,
}


#[derive(Debug, Clone, Deserialize)]
pub struct ToolServer {
    #[serde(default = "default_tool_server_name")]
    pub name: String,
    pub address: String,
    pub port: u16,
    #[serde(default = "default_tool_timeout")]
    pub timeout: u64,
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_base: String,
}

fn default_tool_server_name() -> String { "unnamed-tool-server".into() }
fn default_tool_timeout() -> u64 { 30 }

impl ToolServer {
    pub fn base_url(&self) -> String {
        if self.api_base.is_empty() {
            format!("http://{}:{}", self.address, self.port)
        } else {
            format!("http://{}:{}{}", self.address, self.port, self.api_base)
        }
    }
}

impl std::fmt::Display for ToolServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{} \"{}\" ({}s timeout)", self.address, self.port, self.name, self.timeout)
    }
}

// ── Auth (backward-compat env var bridge) ───────────────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct AuthConfig {
    pub jwt_secret: Option<String>,
    pub jwt_expiry_secs: Option<u64>,
    pub ip_pepper: Option<String>,
    pub users_db: Option<String>,
    pub auth_state_db: Option<String>,
    pub cookie_secure: Option<bool>,
    /// Legacy single-realm fallback. When the realm-specific timeouts
    /// below aren't set, this seeds both realms.
    pub idle_timeout_secs: Option<u64>,
    pub dashboard_idle_timeout_secs: Option<u64>,
    pub webchat_idle_timeout_secs: Option<u64>,
}


// ── Security headers (backward-compat env var bridge) ───────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SecurityConfig {
    pub hsts: Option<String>,
    pub x_content_type_options: Option<String>,
    pub x_frame_options: Option<String>,
    pub csp: Option<String>,
    pub referrer_policy: Option<String>,
    pub permissions_policy: Option<String>,
}


// ── External API (backward-compat env var bridge) ───────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ExternalApiConfig {
    pub mode: Option<String>,
    pub local_endpoint: Option<String>,
    pub remote_endpoint: Option<String>,
    pub remote_api_key: Option<String>,
    pub remote_model: Option<String>,
}


// ── Resources ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ResourcesConfig {
    pub dir: String,
}

impl Default for ResourcesConfig {
    fn default() -> Self {
        Self { dir: "resources".into() }
    }
}

// ── Uploads ─────────────────────────────────────────────────────────
//
// Root directory for per-user, per-chat attachment storage. The
// field is `Option<String>` so "missing from TOML" and "set but
// empty" can be distinguished:
//
//   * `None`       → operator omitted the [uploads] section. Resolve
//                    to `<cwd>/uploads` at startup. Valid.
//   * `Some("")`   → operator explicitly set `dir = ""`. Fatal —
//                    `resolve_uploads_dir` panics.
//   * `Some(path)` → honored verbatim.

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(default)]
pub struct UploadsConfig {
    pub dir: Option<String>,
}

/// Resolve the configured uploads path or derive a default. Panics on
/// an explicit empty string — the goal is to refuse to launch rather
/// than silently accept a misconfiguration that would stash uploads
/// in an unexpected location.
pub fn resolve_uploads_dir(cfg: &UploadsConfig) -> String {
    match cfg.dir.as_deref() {
        Some(s) if s.trim().is_empty() => panic!(
            "ironllm: [uploads] dir is set but empty — either remove the key to use the cwd-derived default, or provide a non-empty path"
        ),
        Some(s) => s.trim().to_string(),
        None => {
            let cwd = std::env::current_dir()
                .expect("ironllm: could not read current working directory to derive uploads dir");
            cwd.join("uploads").to_string_lossy().into_owned()
        }
    }
}

/// Resolve + create the uploads directory tree if it doesn't exist
/// yet. Called at boot so that every downstream handler can trust
/// the path is present. Creation failure is fatal — without an
/// uploads dir the file-attachment feature can't persist anything
/// and startup should abort rather than fail silently on the first
/// upload attempt.
pub fn ensure_uploads_dir(cfg: &UploadsConfig) -> String {
    let dir = resolve_uploads_dir(cfg);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        panic!(
            "ironllm: failed to create uploads dir {dir:?}: {e} — fix the path in [uploads].dir or check filesystem permissions"
        );
    }
    dir
}



// ── Tool executor ───────────────────────────────────────────────────
//
// Endpoints + credentials the `toolexecutor` crate uses to satisfy
// tool-call events fired on the iron-events bus. Two downstream
// services are in play (Crawl4AI, Serper). API keys are intentionally
// `Option<String>` so the operator can supply them in the gitignored
// `ironllm.local.toml` and keep the main config secret-free.
// GraphRAG search is handled natively via the iron-events bus
// (ironllm-rag-search crate) — no HTTP proxy needed.

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ToolExecutorConfig {
    /// `Crawl4AI` base URL (no auth). Used by `crawl_url`, `crawl_and_store`,
    /// `crawl_temp`, `deep_crawl_and_store`.
    pub crawler_url: String,
    /// Full Serper search endpoint URL (e.g. `https://google.serper.dev/search`).
    /// Used by `web_search`.
    pub web_search: String,
    /// Serper API key sent as the `X-API-KEY` header. Sourced from the
    /// operator's `ironllm.local.toml`; `None` disables `web_search`.
    pub web_search_api_key: Option<String>,
}

impl Default for ToolExecutorConfig {
    fn default() -> Self {
        Self {
            crawler_url: "http://localhost:11235".into(),
            web_search: "https://google.serper.dev/search".into(),
            web_search_api_key: None,
        }
    }
}

// ── RAG search (embedding model config) ─────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RagSearchConfig {
    pub embedding_model_name: String,
    pub embedding_model_path: String,
}

impl Default for RagSearchConfig {
    fn default() -> Self {
        Self {
            embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2".into(),
            embedding_model_path: "./embeddingmodel/model.onnx".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RagIngestionConfig {
    pub extraction_url: String,
    pub extraction_model: String,
    pub embedding_model_path: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub min_confidence: f32,
}

impl Default for RagIngestionConfig {
    fn default() -> Self {
        Self {
            extraction_url: "http://127.0.0.1:8078/v1".into(),
            extraction_model: String::new(),
            embedding_model_path: "./embeddingmodel/model.onnx".into(),
            chunk_size: 512,
            chunk_overlap: 50,
            min_confidence: 0.7,
        }
    }
}

// ── Load + env bridge ───────────────────────────────────────────────

/// Load `ironllm.toml` from `path`. Missing file → all defaults.
pub fn load(path: &Path) -> Config {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str::<Config>(&content) {
            Ok(cfg) => {
                eprintln!("ironllm: loaded config from {}", path.display());
                cfg
            }
            Err(e) => {
                eprintln!(
                    "ironllm: TOML parse error in {}: {}\n         → using all defaults",
                    path.display(),
                    e
                );
                Config::default()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "ironllm: config file {} not found → using defaults",
                path.display()
            );
            Config::default()
        }
        Err(e) => {
            eprintln!(
                "ironllm: could not read {}: {} → using defaults",
                path.display(),
                e
            );
            Config::default()
        }
    }
}

/// Load `ironllm.toml` AND apply any overrides from a sibling
/// `ironllm.local.toml` (gitignored, operator-specific). The local file
/// is allowed to carry a subset of fields; only the fields present
/// override the corresponding values from the main file.
///
/// Currently supported override stanzas:
///   * `[test_seed]` → `Config.test_seed`
///
/// Extend `LocalOverrides` below + the merge block to add more.
pub fn load_with_local(path: &Path) -> Config {
    let mut cfg = load(path);

    // Derive ironllm.toml → ironllm.local.toml. Preserves the directory
    // and extension so callers can point `--config` at any path and the
    // local override naming stays consistent.
    let local_path = {
        let parent = path.parent().unwrap_or(Path::new("."));
        let stem = path.file_stem().map_or_else(|| "ironllm".into(), |s| s.to_string_lossy().into_owned());
        let ext = path.extension().map_or_else(|| "toml".into(), |s| s.to_string_lossy().into_owned());
        parent.join(format!("{stem}.local.{ext}"))
    };

    match std::fs::read_to_string(&local_path) {
        Ok(content) => match toml::from_str::<LocalOverrides>(&content) {
            Ok(local) => {
                if let Some(app) = local.application {
                    if let Some(v) = app.claviger_dump { cfg.application.claviger_dump = v; }
                }
                if let Some(a) = local.auth {
                    if a.jwt_secret.is_some() { cfg.auth.jwt_secret = a.jwt_secret; }
                    if a.jwt_expiry_secs.is_some() { cfg.auth.jwt_expiry_secs = a.jwt_expiry_secs; }
                    if a.ip_pepper.is_some() { cfg.auth.ip_pepper = a.ip_pepper; }
                    if a.users_db.is_some() { cfg.auth.users_db = a.users_db; }
                    if a.auth_state_db.is_some() { cfg.auth.auth_state_db = a.auth_state_db; }
                    if a.cookie_secure.is_some() { cfg.auth.cookie_secure = a.cookie_secure; }
                    if a.idle_timeout_secs.is_some() { cfg.auth.idle_timeout_secs = a.idle_timeout_secs; }
                    if a.dashboard_idle_timeout_secs.is_some() { cfg.auth.dashboard_idle_timeout_secs = a.dashboard_idle_timeout_secs; }
                    if a.webchat_idle_timeout_secs.is_some() { cfg.auth.webchat_idle_timeout_secs = a.webchat_idle_timeout_secs; }
                }
                if local.test_seed.is_some() {
                    cfg.test_seed = local.test_seed;
                }
                if local.runtime.is_some() {
                    cfg.runtime = local.runtime;
                }
                if let Some(u) = local.uploads {
                    cfg.uploads = u;
                }
                if let Some(t) = local.toolexecutor {
                    if let Some(v) = t.crawler_url { cfg.toolexecutor.crawler_url = v; }
                    if let Some(v) = t.web_search { cfg.toolexecutor.web_search = v; }
                    if let Some(v) = t.web_search_api_key { cfg.toolexecutor.web_search_api_key = Some(v); }
                }
                if let Some(r) = local.rag_ingestion {
                    if let Some(v) = r.extraction_url { cfg.rag_ingestion.extraction_url = v; }
                    if let Some(v) = r.extraction_model { cfg.rag_ingestion.extraction_model = v; }
                }
                eprintln!(
                    "ironllm: applied local overrides from {}",
                    local_path.display()
                );
            }
            Err(e) => {
                eprintln!(
                    "ironllm: local override parse error in {}: {}\n         → local overrides ignored",
                    local_path.display(),
                    e
                );
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No local file — perfectly normal.
        }
        Err(e) => {
            eprintln!(
                "ironllm: could not read {}: {} → local overrides ignored",
                local_path.display(),
                e
            );
        }
    }

    cfg
}

/// Subset of `Config` that `ironllm.local.toml` may supply. Add fields
/// here as needed; each field is Option so the local file can carry
/// only what it wants to override.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct LocalOverrides {
    application: Option<ApplicationOverride>,
    auth: Option<AuthConfig>,
    test_seed: Option<TestSeedConfig>,
    runtime: Option<RuntimeConfig>,
    uploads: Option<UploadsConfig>,
    toolexecutor: Option<ToolExecutorOverride>,
    rag_ingestion: Option<RagIngestionOverride>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RagIngestionOverride {
    extraction_url: Option<String>,
    extraction_model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ApplicationOverride {
    claviger_dump: Option<bool>,
}

/// All-optional mirror of `ToolExecutorConfig` so the local file can
/// supply just the api keys and inherit the URLs from `ironllm.toml`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ToolExecutorOverride {
    crawler_url: Option<String>,
    web_search: Option<String>,
    web_search_api_key: Option<String>,
}

/// Push TOML values into process env vars so crates that still call
/// `from_env()` pick them up. Only sets a var if it's not already set
/// (CLI env overrides win).
///
/// SAFETY: must be called on the main thread before the tokio runtime
/// spawns any work — same constraint as the old `load_env_file`.
pub unsafe fn populate_env(cfg: &Config) {
    fn set(key: &str, val: &str) {
        if std::env::var_os(key).is_none() {
            unsafe { std::env::set_var(key, val) };
        }
    }

    // Listeners
    set("IRONLLM_BIND_IP", &cfg.application.bind_ip);
    set("IRONLLM_DASHBOARD_PORT", &cfg.dashboard.port.to_string());
    set("IRONLLM_DASHBOARD_API_BASE", &cfg.dashboard.api_base);
    set("IRONLLM_DASHBOARD_DB_PATH", &cfg.dashboard.db_path);
    set("IRONLLM_WEBCHAT_PORT", &cfg.webchat.port.to_string());
    set("IRONLLM_WEBCHAT_API_BASE", &cfg.webchat.api_base);
    set("IRONLLM_WEBCHAT_DB_PATH", &cfg.webchat.db_path);
    set("IRONLLM_ADMIN_PORT", &cfg.admin.port.to_string());
    set("IRONLLM_ADMIN_API_BASE", &cfg.admin.api_base);
    set("IRONLLM_ADMIN_DB_PATH", &cfg.admin.db_path);
    set("IRONLLM_PASSTHROUGH_PORT", &cfg.passthrough.port.to_string());
    set("IRONLLM_PASSTHROUGH_API_BASE", &cfg.passthrough.api_base);
    set("IRONLLM_PASSTHROUGH_DB_PATH", &cfg.passthrough.db_path);

    // LLM targets — flatten back to comma-separated for any legacy reader.
    let csv: String = cfg
        .llm
        .servers
        .iter()
        .map(LlmServer::host_port)
        .collect::<Vec<_>>()
        .join(",");
    set("IRONLLM_LLM_TARGETS", &csv);

    // Auth
    if let Some(v) = &cfg.auth.jwt_secret { set("JWT_SECRET", v); }
    if let Some(v) = cfg.auth.jwt_expiry_secs { set("JWT_EXPIRY_SECS", &v.to_string()); }
    if let Some(v) = &cfg.auth.ip_pepper { set("IRONLLM_IP_PEPPER", v); }
    if let Some(v) = &cfg.auth.users_db { set("IRONLLM_USERS_DB", v); }
    if let Some(v) = &cfg.auth.auth_state_db { set("IRONLLM_AUTH_STATE_DB", v); }
    if let Some(v) = cfg.auth.cookie_secure {
        set("IRONLLM_COOKIE_SECURE", if v { "1" } else { "0" });
    }
    if let Some(v) = cfg.auth.idle_timeout_secs { set("IRONLLM_IDLE_TIMEOUT_SECS", &v.to_string()); }
    if let Some(v) = cfg.auth.dashboard_idle_timeout_secs { set("IRONLLM_DASHBOARD_IDLE_TIMEOUT_SECS", &v.to_string()); }
    if let Some(v) = cfg.auth.webchat_idle_timeout_secs { set("IRONLLM_WEBCHAT_IDLE_TIMEOUT_SECS", &v.to_string()); }

    // Security headers
    if let Some(v) = &cfg.security.hsts { set("IRONLLM_HSTS", v); }
    if let Some(v) = &cfg.security.x_content_type_options { set("IRONLLM_X_CONTENT_TYPE_OPTIONS", v); }
    if let Some(v) = &cfg.security.x_frame_options { set("IRONLLM_X_FRAME_OPTIONS", v); }
    if let Some(v) = &cfg.security.csp { set("IRONLLM_CSP", v); }
    if let Some(v) = &cfg.security.referrer_policy { set("IRONLLM_REFERRER_POLICY", v); }
    if let Some(v) = &cfg.security.permissions_policy { set("IRONLLM_PERMISSIONS_POLICY", v); }

    // External API
    if let Some(v) = &cfg.external_api.mode { set("LLM_MODE", v); }
    if let Some(v) = &cfg.external_api.local_endpoint { set("LLM_LOCAL_ENDPOINT", v); }
    if let Some(v) = &cfg.external_api.remote_endpoint { set("LLM_REMOTE_ENDPOINT", v); }
    if let Some(v) = &cfg.external_api.remote_api_key { set("LLM_REMOTE_API_KEY", v); }
    if let Some(v) = &cfg.external_api.remote_model { set("LLM_REMOTE_MODEL", v); }

    // Resources
    set("IRONLLM_RESOURCES_DIR", &cfg.resources.dir);

    // Uploads — resolve + ensure the tree exists so every crate that
    // reads IRONLLM_UPLOADS_DIR can trust the path is present. Panics
    // on empty string or mkdir failure (see ensure_uploads_dir).
    let uploads_dir = ensure_uploads_dir(&cfg.uploads);
    set("IRONLLM_UPLOADS_DIR", &uploads_dir);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // ── Defaults ─────────────────────────────────────────────────────

    #[test]
    fn config_default_uses_documented_constants() {
        let cfg = Config::default();
        assert_eq!(cfg.application.bind_ip, DEFAULT_BIND_IP);
        assert_eq!(cfg.dashboard.port, DEFAULT_DASHBOARD_PORT);
        assert_eq!(cfg.dashboard.api_base, DEFAULT_DASHBOARD_API_BASE);
        assert_eq!(cfg.dashboard.db_path, DEFAULT_DASHBOARD_DB_PATH);
        assert_eq!(cfg.webchat.port, DEFAULT_WEBCHAT_PORT);
        assert_eq!(cfg.webchat.api_base, DEFAULT_WEBCHAT_API_BASE);
        assert_eq!(cfg.admin.port, DEFAULT_ADMIN_PORT);
        assert_eq!(cfg.passthrough.port, DEFAULT_PASSTHROUGH_PORT);
        assert!(!cfg.tls.enabled);
        assert!(cfg.test_seed.is_none());
    }

    #[test]
    fn llm_config_default_has_one_server_on_8078() {
        let cfg = LlmConfig::default();
        assert_eq!(cfg.servers.len(), 1);
        assert_eq!(cfg.servers[0].address, "127.0.0.1");
        assert_eq!(cfg.servers[0].port, 8078);
        assert_eq!(cfg.servers[0].timeout, DEFAULT_LLM_TIMEOUT);
    }

    #[test]
    fn tools_config_default_is_empty() {
        let cfg = ToolsConfig::default();
        assert!(cfg.servers.is_empty());
    }

    // ── bind_addr: global vs per-listener override ───────────────────

    #[test]
    fn listener_bind_addr_falls_back_to_global() {
        let l = ListenerConfig {
            bind_ip: None,
            port: 1234,
            api_base: "/x".into(),
            db_path: "x".into(),
        };
        assert_eq!(l.bind_addr("0.0.0.0"), "0.0.0.0:1234");
    }

    #[test]
    fn listener_bind_addr_honors_override() {
        let l = ListenerConfig {
            bind_ip: Some("127.0.0.1".into()),
            port: 9999,
            api_base: "/x".into(),
            db_path: "x".into(),
        };
        // Per-listener bind_ip wins over the global.
        assert_eq!(l.bind_addr("0.0.0.0"), "127.0.0.1:9999");
    }

    #[test]
    fn admin_and_passthrough_bind_addr_behave_consistently() {
        let global = "10.0.0.1";
        let admin = AdminListenerConfig::default();
        let pass = PassthroughListenerConfig::default();
        assert_eq!(admin.bind_addr(global), format!("{global}:{DEFAULT_ADMIN_PORT}"));
        assert_eq!(pass.bind_addr(global), format!("{global}:{DEFAULT_PASSTHROUGH_PORT}"));
    }

    // ── LlmServer helpers ────────────────────────────────────────────

    #[test]
    fn llm_server_host_port_concatenates_fields() {
        let s = LlmServer {
            name: "qwen".into(),
            address: "10.1.2.3".into(),
            port: 8080,
            timeout: 60,
        };
        assert_eq!(s.host_port(), "10.1.2.3:8080");
    }

    #[test]
    fn llm_server_display_shape() {
        let s = LlmServer {
            name: "qwen".into(),
            address: "10.1.2.3".into(),
            port: 8080,
            timeout: 60,
        };
        let shown = format!("{s}");
        assert!(shown.contains("10.1.2.3:8080"));
        assert!(shown.contains("qwen"));
        assert!(shown.contains("60s"));
    }

    // ── ToolServer::base_url ─────────────────────────────────────────

    #[test]
    fn tool_server_base_url_without_api_base_omits_path() {
        let t = ToolServer {
            name: "t".into(),
            address: "127.0.0.1".into(),
            port: 7000,
            timeout: 30,
            api_key: None,
            api_base: String::new(),
        };
        assert_eq!(t.base_url(), "http://127.0.0.1:7000");
    }

    #[test]
    fn tool_server_base_url_with_api_base_includes_path() {
        let t = ToolServer {
            name: "t".into(),
            address: "127.0.0.1".into(),
            port: 7000,
            timeout: 30,
            api_key: None,
            api_base: "/v1/tools".into(),
        };
        assert_eq!(t.base_url(), "http://127.0.0.1:7000/v1/tools");
    }

    // ── TOML parsing happy path ──────────────────────────────────────

    #[test]
    fn load_full_config_parses_all_sections() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ironllm.toml");
        std::fs::write(&path, r#"
            [application]
            bind_ip = "10.0.0.5"

            [tls]
            enabled = true
            cert_path = "/etc/ssl/fullchain.pem"
            key_path = "/etc/ssl/privkey.pem"

            [dashboard]
            port = 20000
            api_base = "/d"
            db_path = "/tmp/d.db"

            [webchat]
            port = 20001
            api_base = "/c"
            db_path = "/tmp/c.db"

            [[llm.servers]]
            name = "qwen"
            address = "10.1.1.1"
            port = 9000
            timeout = 120

            [[tools.servers]]
            name = "memtool"
            address = "127.0.0.1"
            port = 7100
            timeout = 20
        "#).unwrap();

        let cfg = load(&path);
        assert_eq!(cfg.application.bind_ip, "10.0.0.5");
        assert!(cfg.tls.enabled);
        assert_eq!(cfg.tls.cert_path, "/etc/ssl/fullchain.pem");
        assert_eq!(cfg.dashboard.port, 20000);
        assert_eq!(cfg.webchat.port, 20001);
        assert_eq!(cfg.llm.servers.len(), 1);
        assert_eq!(cfg.llm.servers[0].name, "qwen");
        assert_eq!(cfg.llm.servers[0].timeout, 120);
        assert_eq!(cfg.tools.servers.len(), 1);
        assert_eq!(cfg.tools.servers[0].port, 7100);
    }

    #[test]
    fn load_partial_config_fills_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("partial.toml");
        std::fs::write(&path, r#"
            [application]
            bind_ip = "127.0.0.1"
        "#).unwrap();

        let cfg = load(&path);
        assert_eq!(cfg.application.bind_ip, "127.0.0.1");
        // Everything else defaults.
        assert_eq!(cfg.webchat.port, DEFAULT_WEBCHAT_PORT);
        assert_eq!(cfg.dashboard.port, DEFAULT_DASHBOARD_PORT);
        assert_eq!(cfg.llm.servers.len(), 1);
    }

    // ── TOML parsing fallback paths ──────────────────────────────────

    #[test]
    fn load_missing_file_returns_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.toml");
        let cfg = load(&path);
        // Should match Config::default()
        assert_eq!(cfg.application.bind_ip, DEFAULT_BIND_IP);
        assert_eq!(cfg.webchat.port, DEFAULT_WEBCHAT_PORT);
    }

    #[test]
    fn load_malformed_toml_returns_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is not = valid = toml [[[[").unwrap();

        let cfg = load(&path);
        // Policy is "on parse error → defaults, not panic".
        assert_eq!(cfg.application.bind_ip, DEFAULT_BIND_IP);
    }

    #[test]
    fn load_wrong_type_returns_defaults() {
        // `port` should be an integer; a string is a type error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("wrong-types.toml");
        std::fs::write(&path, r#"
            [webchat]
            port = "eighty"
        "#).unwrap();

        let cfg = load(&path);
        assert_eq!(cfg.webchat.port, DEFAULT_WEBCHAT_PORT);
    }

    // ── load_with_local: override merge ──────────────────────────────

    #[test]
    fn load_with_local_applies_test_seed_override() {
        let dir = tempdir().unwrap();
        let main = dir.path().join("ironllm.toml");
        let local = dir.path().join("ironllm.local.toml");

        std::fs::write(&main, r#"
            [test_seed]
            password = "<placeholder>"
        "#).unwrap();
        std::fs::write(&local, r#"
            [test_seed]
            password = "real-password-that-is-very-long-47-characters-yes!"
        "#).unwrap();

        let cfg = load_with_local(&main);
        let seed = cfg.test_seed.expect("test_seed should be present");
        assert_eq!(seed.password, "real-password-that-is-very-long-47-characters-yes!");
    }

    #[test]
    fn load_with_local_without_local_file_keeps_main_config() {
        let dir = tempdir().unwrap();
        let main = dir.path().join("ironllm.toml");
        std::fs::write(&main, r#"
            [application]
            bind_ip = "192.168.1.1"

            [test_seed]
            password = "main-pw"
        "#).unwrap();
        // No local file created.

        let cfg = load_with_local(&main);
        assert_eq!(cfg.application.bind_ip, "192.168.1.1");
        assert_eq!(cfg.test_seed.as_ref().unwrap().password, "main-pw");
    }

    #[test]
    fn load_with_local_ignores_malformed_local_file() {
        let dir = tempdir().unwrap();
        let main = dir.path().join("ironllm.toml");
        let local = dir.path().join("ironllm.local.toml");

        std::fs::write(&main, r#"
            [test_seed]
            password = "main-pw"
        "#).unwrap();
        std::fs::write(&local, "{{{ this is not toml").unwrap();

        let cfg = load_with_local(&main);
        // Malformed local is ignored; main values survive.
        assert_eq!(cfg.test_seed.as_ref().unwrap().password, "main-pw");
    }

    // ── LlmServer defaults from serde(default = "...") ───────────────

    #[test]
    fn llm_server_name_defaults_to_unnamed_when_omitted() {
        let toml = r#"
            address = "10.0.0.1"
            port = 8000
        "#;
        let parsed: LlmServer = toml::from_str(toml).unwrap();
        assert_eq!(parsed.name, "unnamed");
        assert_eq!(parsed.timeout, DEFAULT_LLM_TIMEOUT);
    }

    #[test]
    fn llm_server_rejects_missing_address() {
        let toml = r#"
            name = "x"
            port = 8000
        "#;
        let parsed: Result<LlmServer, _> = toml::from_str(toml);
        assert!(parsed.is_err(), "missing address must fail");
    }

    #[test]
    fn llm_server_rejects_missing_port() {
        let toml = r#"
            address = "10.0.0.1"
        "#;
        let parsed: Result<LlmServer, _> = toml::from_str(toml);
        assert!(parsed.is_err(), "missing port must fail");
    }
}
