//! Tool-call stream parser and request builder.
//!
//! OpenAI-compatible LLMs emit tool calls as fragmented SSE deltas:
//!
//! ```text
//! delta.tool_calls[0] = { index: 0, id: "call_abc", function: { name: "search_memory", arguments: "" } }
//! delta.tool_calls[0] = { index: 0, function: { arguments: "{\"ter" } }
//! delta.tool_calls[0] = { index: 0, function: { arguments: "m\": \"vllm\"}" } }
//! ```
//!
//! This module accumulates those fragments into complete `ToolCall`s,
//! then maps each tool name to a ironllmragapi endpoint so the caller
//! can fire the HTTP request.

use serde::Deserialize;
use std::collections::HashMap;

// ── Wire types (deserialized from SSE stream) ───────────────────────

/// One element of `delta.tool_calls[]` in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallDelta {
    /// Positional index — groups fragments belonging to the same call.
    pub index: usize,
    /// Present only on the first fragment for this index.
    #[serde(default)]
    pub id: Option<String>,
    /// Present only on the first fragment.
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// Function name + argument fragments.
    #[serde(default)]
    pub function: Option<ToolCallFunctionDelta>,
}

/// The `function` object inside a tool-call delta.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallFunctionDelta {
    /// Present only on the first fragment.
    #[serde(default)]
    pub name: Option<String>,
    /// Argument JSON fragment — may be empty string on the first chunk,
    /// then subsequent chunks carry the actual characters.
    #[serde(default)]
    pub arguments: Option<String>,
}

// ── Accumulator ─────────────────────────────────────────────────────

/// A fully-assembled tool call ready for execution.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The `call_*` ID assigned by the LLM — must be echoed back in
    /// the `tool` role message so the model can match response to call.
    pub id: String,
    /// Function name (e.g. `"search_memory"`).
    pub name: String,
    /// Complete JSON arguments string.
    pub arguments: String,
}

/// Collects streamed `ToolCallDelta` fragments and assembles them into
/// complete `ToolCall`s once the stream signals `finish_reason: "tool_calls"`.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    /// In-flight calls keyed by their positional index.
    pending: HashMap<usize, PendingCall>,
}

#[derive(Debug, Default)]
struct PendingCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one delta fragment into the accumulator.
    pub fn push(&mut self, delta: &ToolCallDelta) {
        let entry = self.pending.entry(delta.index).or_default();

        if let Some(id) = &delta.id {
            entry.id = Some(id.clone());
        }

        if let Some(func) = &delta.function {
            if let Some(name) = &func.name {
                entry.name = Some(name.clone());
            }
            if let Some(args) = &func.arguments {
                entry.arguments.push_str(args);
            }
        }
    }

    /// Returns true if any fragments have been accumulated.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Drain all accumulated fragments into finished `ToolCall`s.
    /// Call this when `finish_reason == "tool_calls"`.
    ///
    /// Calls missing an id or name are logged and skipped.
    pub fn finish(&mut self) -> Vec<ToolCall> {
        let mut calls = Vec::with_capacity(self.pending.len());

        // Drain sorted by index so execution order is deterministic.
        let mut indices: Vec<usize> = self.pending.keys().copied().collect();
        indices.sort_unstable();

        for idx in indices {
            if let Some(pending) = self.pending.remove(&idx) {
                match (pending.id, pending.name) {
                    (Some(id), Some(name)) => {
                        calls.push(ToolCall {
                            id,
                            name,
                            arguments: pending.arguments,
                        });
                    }
                    (id, name) => {
                        tracing::warn!(
                            index = idx,
                            id = ?id,
                            name = ?name,
                            "Dropping incomplete tool call — missing id or name"
                        );
                    }
                }
            }
        }

        calls
    }

    /// Discard all state (e.g. on stream error).
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

// ── JSON healing ────────────────────────────────────────────────────

/// Attempt to repair truncated JSON by closing unterminated strings,
/// objects, and arrays.  The LLM sometimes runs out of tokens mid-
/// argument, producing e.g. `{"query": "some search term` with no
/// closing `"}`.  Returns `Some(healed)` on success, `None` if the
/// input is beyond simple repair.
pub fn heal_json(raw: &str) -> Option<String> {
    // Already valid — nothing to do.
    if serde_json::from_str::<serde_json::Value>(raw).is_ok() {
        return Some(raw.to_string());
    }

    // Walk the raw text to figure out what was left open.
    let mut in_string = false;
    let mut escape_next = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in raw.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }

    // Try closing everything that is open.
    let mut healed = raw.to_string();
    if in_string {
        healed.push('"');
    }
    for closer in stack.into_iter().rev() {
        healed.push(closer);
    }

    if serde_json::from_str::<serde_json::Value>(&healed).is_ok() {
        return Some(healed);
    }

    // Simple closure failed (e.g. truncated mid-key like `{"query": "x", "nu`).
    // Trim trailing characters until closure produces valid JSON.
    let len = raw.len();
    for trim in 1..len.min(200) {
        let candidate = &raw[..len - trim];
        let mut in_str = false;
        let mut esc = false;
        let mut stk: Vec<char> = Vec::new();

        for ch in candidate.chars() {
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                if ch == '\\' {
                    esc = true;
                    continue;
                }
                if ch == '"' {
                    in_str = false;
                }
                continue;
            }
            match ch {
                '"' => in_str = true,
                '{' => stk.push('}'),
                '[' => stk.push(']'),
                '}' | ']' => {
                    stk.pop();
                }
                _ => {}
            }
        }

        let mut test = candidate.to_string();
        if in_str {
            test.push('"');
        }
        for c in stk.into_iter().rev() {
            test.push(c);
        }

        if serde_json::from_str::<serde_json::Value>(&test).is_ok() {
            return Some(test);
        }
    }

    None
}

// ── Route table ─────────────────────────────────────────────────────

/// HTTP method for a tool endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

/// A resolved route: method + path suffix (appended to `ToolServer::base_url()`).
#[derive(Debug, Clone)]
pub struct ToolRoute {
    pub method: Method,
    /// Path relative to the tool server's `api_base`.
    /// e.g. `"/search"` → full URL becomes `http://127.0.0.1:8081/api/v2/search`.
    pub path: &'static str,
}

/// A fully resolved request ready to be sent by the executor.
#[derive(Debug)]
pub struct ToolRequest {
    /// The tool call ID — echoed back to the LLM in the tool-role message.
    pub call_id: String,
    /// Tool function name (for status messages / logging).
    pub tool_name: String,
    /// Full URL to call (e.g. `http://127.0.0.1:8081/api/v2/search`).
    pub url: String,
    /// HTTP method.
    pub method: Method,
    /// Parsed JSON body — `None` for GET requests, `Some(...)` for POST.
    pub body: Option<serde_json::Value>,
    /// Optional bearer token from the tool server config.
    pub bearer_token: Option<String>,
    /// Timeout for this request.
    pub timeout: std::time::Duration,
}

/// Build a `ToolRequest` from a completed `ToolCall`, a route the
/// caller has already resolved, and the tool-server config.
///
/// The legacy `route_for` lookup is gone — every supported tool now
/// dispatches via the `iron-events` bus path in `toolexecutor`. This
/// function is kept as a transport helper so a future operator who
/// wires up a *new* HTTP tool server can compose it with their own
/// route resolver (registry, config-driven map, etc.) without
/// touching this crate.
///
/// Returns `Err` with a human-readable message if the call's
/// arguments JSON is malformed (POST routes only).
pub fn build_request(
    call: &ToolCall,
    route: &ToolRoute,
    server_base_url: &str,
    api_key: Option<&str>,
    _timeout_secs: u64,
) -> Result<ToolRequest, String> {
    let url = format!("{}{}", server_base_url, route.path);

    let body = match route.method {
        Method::Get => None,
        Method::Post => {
            let parsed: serde_json::Value = serde_json::from_str(&call.arguments)
                .map_err(|e| format!(
                    "Invalid arguments JSON for {}: {} — raw: {}",
                    call.name, e, call.arguments
                ))?;
            Some(parsed)
        }
    };

    Ok(ToolRequest {
        call_id: call.id.clone(),
        tool_name: call.name.clone(),
        url,
        method: route.method,
        body,
        bearer_token: api_key.map(std::borrow::ToOwned::to_owned),
        // DIAGNOSTIC: override per-tool-request timeout to 10_000 s
        // (configured `timeout_secs` ignored) to rule out tool-request
        // timeouts as the cause of the client SSE disconnects.
        timeout: std::time::Duration::from_secs(10_000),
    })
}

// ── Status messages ─────────────────────────────────────────────────

/// Human-readable status text for a tool call, including key arguments.
/// `args` should be the parsed JSON arguments (best-effort — falls
/// back to generic text if parsing failed upstream).
pub fn status_text(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "search_memory" => {
            let term = args["term"].as_str().unwrap_or("...");
            let depth = args["depth"].as_str().unwrap_or("medium");
            format!("Searching memory for \"{term}\" (depth: {depth})")
        }
        "crawl_url" => {
            let url = args["url"].as_str().unwrap_or("...");
            let max = args["max_chars"].as_i64().unwrap_or(5000);
            format!("Crawling {url} (max {max} chars)")
        }
        "save_url" => {
            let url = args["url"].as_str().unwrap_or("...");
            format!("Saving {url} to memory")
        }
        "scrape_site" => {
            let url = args["url"].as_str().unwrap_or("...");
            let depth = args["max_depth"].as_i64().unwrap_or(2);
            let pages = args["max_pages"].as_i64().unwrap_or(10);
            format!("Deep-crawling {url} (depth: {depth}, max {pages} pages)")
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("...");
            let n = args["num_results"].as_i64().unwrap_or(10);
            format!("Searching the web for \"{query}\" ({n} results)")
        }
        "memory_stats"  => "Checking memory stats".to_string(),
        "system_health" => "Checking system health".to_string(),

        // Coder sandbox tools. Bash carries a `description` parameter in its
        // schema; the rest identify themselves by path or pattern.
        "Bash" => {
            let desc = args["description"].as_str();
            let cmd  = args["command"].as_str();
            match (desc, cmd) {
                (Some(d), _)    => format!("Executing tool: Bash (\"description\": \"{d}\")"),
                (None, Some(c)) => {
                    let preview: String = c.chars().take(80).collect();
                    format!("Executing tool: Bash (\"command\": \"{preview}\")")
                }
                _ => "Executing tool: Bash".to_string(),
            }
        }
        "Read" => {
            let path = args["file_path"].as_str().unwrap_or("...");
            format!("Executing tool: Read (\"file_path\": \"{path}\")")
        }
        "Write" => {
            let path = args["file_path"].as_str().unwrap_or("...");
            format!("Executing tool: Write (\"file_path\": \"{path}\")")
        }
        "Edit" => {
            let path = args["file_path"].as_str().unwrap_or("...");
            format!("Executing tool: Edit (\"file_path\": \"{path}\")")
        }
        "Glob" => {
            let pattern = args["pattern"].as_str().unwrap_or("...");
            format!("Executing tool: Glob (\"pattern\": \"{pattern}\")")
        }
        "Grep" => {
            let pattern = args["pattern"].as_str().unwrap_or("...");
            format!("Executing tool: Grep (\"pattern\": \"{pattern}\")")
        }
        "PresentFile" => {
            let path = args["file_path"].as_str().unwrap_or("...");
            format!("Presenting file from /workspace/outputs/: {path}")
        }

        _ => format!("Executing tool: {tool_name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_single_call() {
        let mut acc = ToolCallAccumulator::new();

        // First fragment: id + name + empty args
        acc.push(&ToolCallDelta {
            index: 0,
            id: Some("call_abc123".into()),
            kind: Some("function".into()),
            function: Some(ToolCallFunctionDelta {
                name: Some("search_memory".into()),
                arguments: Some(String::new()),
            }),
        });

        // Argument fragments
        acc.push(&ToolCallDelta {
            index: 0,
            id: None,
            kind: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some("{\"ter".into()),
            }),
        });

        acc.push(&ToolCallDelta {
            index: 0,
            id: None,
            kind: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some("m\": \"vllm\"}".into()),
            }),
        });

        assert!(acc.has_pending());

        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc123");
        assert_eq!(calls[0].name, "search_memory");
        assert_eq!(calls[0].arguments, "{\"term\": \"vllm\"}");
        assert!(!acc.has_pending());
    }

    #[test]
    fn accumulate_parallel_calls() {
        let mut acc = ToolCallAccumulator::new();

        // Two calls interleaved
        acc.push(&ToolCallDelta {
            index: 0,
            id: Some("call_1".into()),
            kind: Some("function".into()),
            function: Some(ToolCallFunctionDelta {
                name: Some("web_search".into()),
                arguments: Some(String::new()),
            }),
        });
        acc.push(&ToolCallDelta {
            index: 1,
            id: Some("call_2".into()),
            kind: Some("function".into()),
            function: Some(ToolCallFunctionDelta {
                name: Some("memory_stats".into()),
                arguments: Some(String::new()),
            }),
        });
        acc.push(&ToolCallDelta {
            index: 0,
            id: None,
            kind: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some("{\"query\": \"rust\"}".into()),
            }),
        });
        acc.push(&ToolCallDelta {
            index: 1,
            id: None,
            kind: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some("{}".into()),
            }),
        });

        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        // Sorted by index
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "memory_stats");
    }

    #[test]
    fn build_post_request() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "search_memory".into(),
            arguments: r#"{"term": "vllm", "depth": "high"}"#.into(),
        };
        let route = ToolRoute { method: Method::Post, path: "/search" };

        let req = build_request(&call, &route, "http://127.0.0.1:8081/api/v2", Some("tok123"), 30)
            .expect("should succeed");

        assert_eq!(req.url, "http://127.0.0.1:8081/api/v2/search");
        assert_eq!(req.method, Method::Post);
        assert!(req.body.is_some());
        assert_eq!(req.bearer_token.as_deref(), Some("tok123"));
        assert_eq!(req.call_id, "call_1");
    }

    #[test]
    fn build_get_request() {
        let call = ToolCall {
            id: "call_2".into(),
            name: "memory_stats".into(),
            arguments: "{}".into(),
        };
        let route = ToolRoute { method: Method::Get, path: "/stats" };

        let req = build_request(&call, &route, "http://127.0.0.1:8081/api/v2", None, 30)
            .expect("should succeed");

        assert_eq!(req.url, "http://127.0.0.1:8081/api/v2/stats");
        assert_eq!(req.method, Method::Get);
        assert!(req.body.is_none());
        assert!(req.bearer_token.is_none());
    }

    #[test]
    fn build_request_bad_json() {
        let call = ToolCall {
            id: "call_y".into(),
            name: "search_memory".into(),
            arguments: "not valid json".into(),
        };
        let route = ToolRoute { method: Method::Post, path: "/search" };
        assert!(build_request(&call, &route, "http://localhost/api/v2", None, 30).is_err());
    }

    // ── heal_json tests ────────────────────────────────────────────

    #[test]
    fn heal_already_valid() {
        let input = r#"{"query": "rust async patterns"}"#;
        assert_eq!(heal_json(input).unwrap(), input);
    }

    #[test]
    fn heal_unterminated_string_value() {
        // Real pattern from ironllm.log line 365:
        // {"query": "\"Opus 4.6\" benchmark scores
        let input = r#"{"query": "\"Opus 4.6\" benchmark scores"#;
        let healed = heal_json(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&healed).unwrap();
        assert!(parsed["query"].as_str().unwrap().contains("Opus 4.6"));
    }

    #[test]
    fn heal_unterminated_long_query() {
        // Real pattern from ironllm.log line 570:
        // {"query": "Claude Opus 4.6 MMLU-Pro MATH benchmark scores official
        let input = r#"{"query": "Claude Opus 4.6 MMLU-Pro MATH benchmark scores official"#;
        let healed = heal_json(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&healed).unwrap();
        assert!(parsed["query"].as_str().unwrap().contains("MMLU-Pro"));
    }

    #[test]
    fn heal_string_closed_but_object_open() {
        // LLM closed the string but ran out before closing the brace.
        let input = r#"{"query": "some search""#;
        let healed = heal_json(input).unwrap();
        serde_json::from_str::<serde_json::Value>(&healed).unwrap();
    }

    #[test]
    fn heal_truncated_mid_second_key() {
        // LLM started a second key then got cut off:
        // {"query": "test", "num
        let input = r#"{"query": "test", "num"#;
        let healed = heal_json(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&healed).unwrap();
        // Should at least preserve the first complete key.
        assert_eq!(parsed["query"].as_str().unwrap(), "test");
    }

    #[test]
    fn heal_truncated_mid_second_value() {
        // {"query": "test", "num_results": 1
        // (missing closing brace)
        let input = r#"{"query": "test", "num_results": 1"#;
        let healed = heal_json(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&healed).unwrap();
        assert_eq!(parsed["query"].as_str().unwrap(), "test");
        assert_eq!(parsed["num_results"].as_i64().unwrap(), 1);
    }

    #[test]
    fn heal_nested_array_truncated() {
        let input = r#"{"tags": ["a", "b"#;
        let healed = heal_json(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&healed).unwrap();
        assert_eq!(parsed["tags"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn heal_empty_string() {
        // Empty input is not recoverable to a meaningful object.
        assert!(heal_json("").is_none() || heal_json("").unwrap() == "");
    }

    #[test]
    fn heal_total_garbage() {
        // Completely unparseable — should return None.
        assert!(heal_json("not json at all").is_none());
    }
}
