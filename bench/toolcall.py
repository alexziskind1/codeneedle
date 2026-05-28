"""Tool calling benchmark — validates LLM tool call generation accuracy."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── Tool definitions (OpenAI format) ─────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    # ── Agentic ──
    {"type": "function", "function": {"name": "search_memory", "description": "Search the GraphRAG knowledge graph using hybrid vector + graph search.", "parameters": {"type": "object", "properties": {"term": {"type": "string", "description": "Search query or topic"}, "depth": {"type": "string", "description": "Search depth level", "enum": ["low", "medium", "high", "all"], "default": "medium"}, "limit": {"type": "integer", "description": "Max chunks to return (5-50)", "default": 10, "minimum": 5, "maximum": 50}}, "required": ["term"]}}},
    {"type": "function", "function": {"name": "crawl_url", "description": "Fetch web page content from a URL as cleaned markdown.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Target URL to crawl"}, "max_chars": {"type": "integer", "description": "Max characters to return (5000-25000)", "default": 5000, "minimum": 5000, "maximum": 25000}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "save_url", "description": "Crawl a URL and store content in the GraphRAG database.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Target URL to crawl"}, "retention_policy": {"type": "string", "description": "Storage retention policy", "enum": ["permanent", "session_only", "30_days"], "default": "permanent"}, "tags": {"type": "string", "description": "Comma-separated tags", "default": ""}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "scrape_site", "description": "BFS deep crawl with language filtering. Follows links up to max_depth.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Starting URL"}, "retention_policy": {"type": "string", "description": "Storage retention", "enum": ["permanent", "session_only", "30_days"], "default": "permanent"}, "tags": {"type": "string", "description": "Comma-separated tags", "default": ""}, "max_depth": {"type": "integer", "description": "Max crawl depth (1-5)", "default": 2, "minimum": 1, "maximum": 5}, "max_pages": {"type": "integer", "description": "Max pages to store (1-250)", "default": 10, "minimum": 1, "maximum": 250}, "include_external": {"type": "boolean", "description": "Follow external links", "default": False}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web. Returns titles, URLs, and snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query string"}, "num_results": {"type": "integer", "description": "Max results (1-20)", "default": 10, "minimum": 1, "maximum": 20}, "max_chars_per_result": {"type": "integer", "description": "Max chars per snippet (200-15000)", "default": 500, "minimum": 200, "maximum": 15000}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "memory_stats", "description": "Get statistics from the knowledge graph database.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "system_health", "description": "Get health status of all Docker containers.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    # ── CC Base ──
    {"type": "function", "function": {"name": "Agent", "description": "Launch a new agent for complex multi-step tasks.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Task for the agent"}, "description": {"type": "string", "description": "Short summary"}, "subagent_type": {"type": "string", "description": "Specialized agent type"}, "isolation": {"type": "string", "description": "Isolation mode", "enum": ["worktree"]}, "run_in_background": {"type": "boolean", "description": "Run in background", "default": False}, "model": {"type": "string", "description": "Model override", "enum": ["sonnet", "opus", "haiku"]}}, "required": ["prompt"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Execute a bash command and return output.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Command to execute"}, "description": {"type": "string", "description": "What the command does"}, "timeout": {"type": "number", "description": "Timeout in ms (max 600000)", "maximum": 600000}, "run_in_background": {"type": "boolean", "description": "Run in background", "default": False}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Read", "description": "Read a file from the filesystem.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to file"}, "offset": {"type": "number", "description": "Line number to start from"}, "limit": {"type": "number", "description": "Number of lines to read", "default": 2000}, "pages": {"type": "string", "description": "Page range for PDFs"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Exact string replacement in files.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to file"}, "old_string": {"type": "string", "description": "Text to replace"}, "new_string": {"type": "string", "description": "Replacement text"}, "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False}}, "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Write a file to the filesystem.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path to file"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "Glob", "description": "Fast file pattern matching.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern"}, "path": {"type": "string", "description": "Directory to search"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search files with regex (ripgrep).", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern"}, "path": {"type": "string", "description": "File or directory"}, "glob": {"type": "string", "description": "File glob filter"}, "type": {"type": "string", "description": "File type filter"}, "output_mode": {"type": "string", "description": "Output mode", "enum": ["content", "files_with_matches", "count"], "default": "files_with_matches"}, "-i": {"type": "boolean", "description": "Case insensitive", "default": False}, "-n": {"type": "boolean", "description": "Show line numbers", "default": True}, "-A": {"type": "number", "description": "Lines after match"}, "-B": {"type": "number", "description": "Lines before match"}, "-C": {"type": "number", "description": "Context lines"}, "multiline": {"type": "boolean", "description": "Multiline mode", "default": False}, "head_limit": {"type": "number", "description": "Limit output entries", "default": 0}, "offset": {"type": "number", "description": "Skip entries", "default": 0}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "PresentFile", "description": "Present a file to the user as a download.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path under /workspace/outputs/"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "WebFetch", "description": "Fetch and process content from a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL to fetch"}, "prompt": {"type": "string", "description": "What to extract from the page"}}, "required": ["url", "prompt"]}}},
    {"type": "function", "function": {"name": "NotebookEdit", "description": "Edit Jupyter notebook cells.", "parameters": {"type": "object", "properties": {"notebook_path": {"type": "string", "description": "Path to .ipynb file"}, "cell_number": {"type": "integer", "description": "0-indexed cell position"}, "new_source": {"type": "string", "description": "New cell content"}, "edit_mode": {"type": "string", "description": "Operation", "enum": ["replace", "insert", "delete"], "default": "replace"}}, "required": ["notebook_path", "cell_number", "new_source"]}}},
    {"type": "function", "function": {"name": "AskUserQuestion", "description": "Ask the user a multiple-choice question.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The question to ask"}, "options": {"type": "array", "description": "Array of choice objects with header, label, description"}, "multiSelect": {"type": "boolean", "description": "Allow multiple selections", "default": False}}, "required": ["question", "options"]}}},
    {"type": "function", "function": {"name": "LSP", "description": "Language Server Protocol operations for code intelligence.", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "description": "LSP operation", "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol", "workspaceSymbol", "goToImplementation", "prepareCallHierarchy", "incomingCalls", "outgoingCalls"]}, "file_path": {"type": "string", "description": "Path to file"}, "line": {"type": "integer", "description": "Line number (1-based)"}, "character": {"type": "integer", "description": "Character offset (1-based)"}}, "required": ["operation", "file_path", "line", "character"]}}},
    {"type": "function", "function": {"name": "Skill", "description": "Execute a skill (slash command).", "parameters": {"type": "object", "properties": {"skill": {"type": "string", "description": "Skill name"}, "args": {"type": "string", "description": "Optional arguments"}}, "required": ["skill"]}}},
    {"type": "function", "function": {"name": "SearchConversation", "description": "Search prior user messages with regex.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern"}, "-i": {"type": "boolean", "description": "Case insensitive", "default": False}, "context": {"type": "number", "description": "Surrounding lines", "default": 0}, "limit": {"type": "number", "description": "Max matches", "default": 50}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "SendMessage", "description": "Send a message to a teammate or peer agent.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "Recipient name or address"}, "message": {"type": "string", "description": "Message content"}, "summary": {"type": "string", "description": "Short preview"}}, "required": ["to", "message"]}}},
    {"type": "function", "function": {"name": "SendUserMessage", "description": "Send a message to the user with optional attachments.", "parameters": {"type": "object", "properties": {"message": {"type": "string", "description": "Markdown content"}, "attachments": {"type": "array", "description": "File paths"}, "status": {"type": "string", "description": "Message type", "enum": ["normal", "proactive"], "default": "normal"}}, "required": ["message"]}}},
]

# ── Schema lookup ────────────────────────────────────────────────────

_SCHEMA_BY_NAME: dict[str, dict] = {}


def _get_schema(name: str) -> dict | None:
    if not _SCHEMA_BY_NAME:
        for t in TOOL_DEFINITIONS:
            _SCHEMA_BY_NAME[t["function"]["name"]] = t["function"]["parameters"]
    return _SCHEMA_BY_NAME.get(name)


# ── Mock responses ───────────────────────────────────────────────────

MOCK_RESPONSES: dict[str, str] = {
    "search_memory": '{"results": [{"chunk": "Relevant data found.", "score": 0.92}]}',
    "crawl_url": '{"content": "# Page Title\\nPage content here.", "chars": 1200}',
    "save_url": '{"status": "saved", "chunks": 3}',
    "scrape_site": '{"status": "complete", "pages_stored": 5}',
    "web_search": '{"results": [{"title": "Result 1", "url": "https://example.com", "snippet": "Info."}]}',
    "memory_stats": '{"total_chunks": 142, "total_entities": 89, "total_documents": 12}',
    "system_health": '{"containers": [{"name": "vllm", "status": "running"}, {"name": "neo4j", "status": "running"}]}',
    "Agent": '{"result": "Task completed successfully.", "agent_id": "a1b2c3"}',
    "Bash": '{"stdout": "OK", "exit_code": 0}',
    "Read": '{"content": "1\\tconst x = 1;\\n2\\tconst y = 2;"}',
    "Edit": '{"status": "success", "replacements": 1}',
    "Write": '{"status": "success", "bytes_written": 256}',
    "Glob": '{"files": ["src/main.rs", "src/lib.rs"]}',
    "Grep": '{"matches": ["src/lib.rs:42: fn main()"]}',
    "PresentFile": '{"status": "presented"}',
    "WebFetch": '{"content": "Extracted page summary."}',
    "NotebookEdit": '{"status": "cell updated"}',
    "AskUserQuestion": '{"answer": "Option A selected"}',
    "LSP": '{"result": [{"file": "src/lib.rs", "line": 42, "character": 5}]}',
    "Skill": '{"status": "skill executed"}',
    "SearchConversation": '{"matches": [{"turn": 3, "text": "matched line"}]}',
    "SendMessage": '{"status": "sent"}',
    "SendUserMessage": '{"status": "delivered"}',
}

# ── Prompt variants ──────────────────────────────────────────────────


@dataclass
class ToolTest:
    prompt: str
    expected_tool: str
    required_args: dict[str, type | None] = field(default_factory=dict)
    expected_args: dict[str, object] | None = None


TOOL_TESTS: list[ToolTest] = [
    # ── search_memory (10) ──
    ToolTest("Search my memory for vLLM configuration tips", "search_memory", {"term": str}),
    ToolTest("Look up what I have saved about React hooks", "search_memory", {"term": str}),
    ToolTest("Do a deep search in my knowledge base for CUDA optimization", "search_memory", {"term": str}, {"depth": "high"}),
    ToolTest("Quick shallow search for python async patterns", "search_memory", {"term": str}, {"depth": "low"}),
    ToolTest("Search memory for Docker networking, return 30 results", "search_memory", {"term": str}, {"limit": 30}),
    ToolTest("Find everything in my knowledge graph about ROCm drivers", "search_memory", {"term": str}, {"depth": "all"}),
    ToolTest("Search for Rust lifetime annotations, just 5 results", "search_memory", {"term": str}, {"limit": 5}),
    ToolTest("Check my memory for any info on PostgreSQL indexing strategies", "search_memory", {"term": str}),
    ToolTest("Deep dive search for transformer attention mechanisms, give me 50 results", "search_memory", {"term": str}, {"depth": "high", "limit": 50}),
    ToolTest("Do a medium depth search for WebSocket implementation patterns", "search_memory", {"term": str}, {"depth": "medium"}),

    # ── crawl_url (10) ──
    ToolTest("Fetch the content from https://docs.rs/tokio/latest", "crawl_url", {"url": str}),
    ToolTest("Crawl https://react.dev/learn and get me the page content", "crawl_url", {"url": str}),
    ToolTest("Get the full page from https://example.com/article, I need up to 20000 characters", "crawl_url", {"url": str}, {"max_chars": 20000}),
    ToolTest("Fetch https://blog.rust-lang.org/2024/01/post with the default character limit", "crawl_url", {"url": str}),
    ToolTest("Read the content at https://docs.python.org/3/library/asyncio.html, max 10000 chars", "crawl_url", {"url": str}, {"max_chars": 10000}),
    ToolTest("Crawl https://en.wikipedia.org/wiki/Neural_network", "crawl_url", {"url": str}),
    ToolTest("Grab the page at https://arxiv.org/abs/2301.00001, give me 25000 chars", "crawl_url", {"url": str}, {"max_chars": 25000}),
    ToolTest("Fetch https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "crawl_url", {"url": str}),
    ToolTest("Get content from https://huggingface.co/docs/transformers, 15000 char limit", "crawl_url", {"url": str}, {"max_chars": 15000}),
    ToolTest("Crawl https://www.kernel.org/doc/html/latest/", "crawl_url", {"url": str}),

    # ── save_url (10) ──
    ToolTest("Save https://docs.rs/axum/latest to my knowledge base", "save_url", {"url": str}),
    ToolTest("Store https://react.dev/reference permanently with tags 'frontend,react'", "save_url", {"url": str}, {"retention_policy": "permanent", "tags": "frontend,react"}),
    ToolTest("Save https://example.com/temp-doc for this session only", "save_url", {"url": str}, {"retention_policy": "session_only"}),
    ToolTest("Archive https://blog.cloudflare.com/post into my database", "save_url", {"url": str}),
    ToolTest("Store https://docs.nvidia.com/cuda with 30 day retention", "save_url", {"url": str}, {"retention_policy": "30_days"}),
    ToolTest("Save https://pytorch.org/docs/stable/index.html tagged as 'ml,pytorch'", "save_url", {"url": str}),
    ToolTest("Add https://doc.rust-lang.org/book to my memory permanently", "save_url", {"url": str}),
    ToolTest("Save https://redis.io/docs for the session, tag it 'database'", "save_url", {"url": str}, {"retention_policy": "session_only", "tags": "database"}),
    ToolTest("Store https://github.com/tokio-rs/tokio/wiki in the knowledge graph", "save_url", {"url": str}),
    ToolTest("Save this page https://www.postgresql.org/docs/16/index.html with tags 'postgres,db'", "save_url", {"url": str}),

    # ── scrape_site (10) ──
    ToolTest("Deep crawl https://docs.rs/hyper going 3 levels deep", "scrape_site", {"url": str}, {"max_depth": 3}),
    ToolTest("Scrape https://react.dev, store up to 50 pages", "scrape_site", {"url": str}, {"max_pages": 50}),
    ToolTest("Crawl https://docs.python.org/3/ with depth 2 and include external links", "scrape_site", {"url": str}, {"max_depth": 2, "include_external": True}),
    ToolTest("Scrape https://kubernetes.io/docs, max 100 pages, tag as 'k8s'", "scrape_site", {"url": str}, {"max_pages": 100, "tags": "k8s"}),
    ToolTest("Deep crawl https://rocm.docs.amd.com to depth 5, store 250 pages", "scrape_site", {"url": str}, {"max_depth": 5, "max_pages": 250}),
    ToolTest("Scrape https://developer.mozilla.org/en-US/docs/Web, session only retention", "scrape_site", {"url": str}, {"retention_policy": "session_only"}),
    ToolTest("Crawl https://docs.docker.com with default settings", "scrape_site", {"url": str}),
    ToolTest("Scrape https://go.dev/doc, 1 level deep, 20 pages max", "scrape_site", {"url": str}, {"max_depth": 1, "max_pages": 20}),
    ToolTest("Deep crawl https://www.typescriptlang.org/docs, keep for 30 days", "scrape_site", {"url": str}, {"retention_policy": "30_days"}),
    ToolTest("Scrape https://ziglang.org/documentation, depth 4, no external links, tag 'zig'", "scrape_site", {"url": str}, {"max_depth": 4, "include_external": False, "tags": "zig"}),

    # ── web_search (10) ──
    ToolTest("Search the web for 'Rust async runtime comparison 2024'", "web_search", {"query": str}),
    ToolTest("Google for ROCm 7.2 release notes", "web_search", {"query": str}),
    ToolTest("Web search for vLLM tensor parallelism guide, give me 5 results", "web_search", {"query": str}, {"num_results": 5}),
    ToolTest("Search for 'RDNA4 compute shader performance' with 20 results", "web_search", {"query": str}, {"num_results": 20}),
    ToolTest("Look up PyTorch 2.5 new features online", "web_search", {"query": str}),
    ToolTest("Search the internet for Cloudflare Workers KV documentation", "web_search", {"query": str}),
    ToolTest("Web search for 'transformer attention is all you need paper', max 3 results with long snippets of 5000 chars", "web_search", {"query": str}, {"num_results": 3, "max_chars_per_result": 5000}),
    ToolTest("Find info online about AMD MI300X benchmarks", "web_search", {"query": str}),
    ToolTest("Search for recent Linux kernel 6.x performance improvements", "web_search", {"query": str}),
    ToolTest("Google for 'neo4j graph database vector search setup guide'", "web_search", {"query": str}),

    # ── memory_stats (10) ──
    ToolTest("Show me my knowledge base statistics", "memory_stats", {}),
    ToolTest("How much data is in my memory?", "memory_stats", {}),
    ToolTest("What's stored in the knowledge graph?", "memory_stats", {}),
    ToolTest("Give me the database stats", "memory_stats", {}),
    ToolTest("How many documents have been ingested?", "memory_stats", {}),
    ToolTest("Show memory usage and entity counts", "memory_stats", {}),
    ToolTest("What domains are in my knowledge base?", "memory_stats", {}),
    ToolTest("Check how many chunks are stored", "memory_stats", {}),
    ToolTest("Display the knowledge graph statistics overview", "memory_stats", {}),
    ToolTest("What's the current state of my saved knowledge?", "memory_stats", {}),

    # ── system_health (10) ──
    ToolTest("Check if all containers are running", "system_health", {}),
    ToolTest("What's the health status of the system?", "system_health", {}),
    ToolTest("Are all Docker services up?", "system_health", {}),
    ToolTest("Show me container status", "system_health", {}),
    ToolTest("Is everything healthy?", "system_health", {}),
    ToolTest("Check system health", "system_health", {}),
    ToolTest("Status of all running services", "system_health", {}),
    ToolTest("Any containers down?", "system_health", {}),
    ToolTest("Give me a health check on the infrastructure", "system_health", {}),
    ToolTest("Docker container health report", "system_health", {}),

    # ── Agent (10) ──
    ToolTest("Spin up an agent to research the best Rust web frameworks", "Agent", {"prompt": str}),
    ToolTest("Launch a background agent to audit the security of src/auth.rs", "Agent", {"prompt": str}, {"run_in_background": True}),
    ToolTest("Create an agent using the opus model to refactor the database layer", "Agent", {"prompt": str}, {"model": "opus"}),
    ToolTest("Start an agent in a worktree to experiment with the new API design", "Agent", {"prompt": str}, {"isolation": "worktree"}),
    ToolTest("Spawn an agent to find all TODO comments in the codebase", "Agent", {"prompt": str}),
    ToolTest("Launch a haiku agent to summarize the README files", "Agent", {"prompt": str}, {"model": "haiku"}),
    ToolTest("Create an agent to run the test suite and report failures", "Agent", {"prompt": str}),
    ToolTest("Start a background agent with sonnet to optimize the build pipeline", "Agent", {"prompt": str}, {"run_in_background": True, "model": "sonnet"}),
    ToolTest("Spin up an explorer agent to map the project structure", "Agent", {"prompt": str}, {"subagent_type": "Explore"}),
    ToolTest("Launch an agent to investigate the memory leak in the worker pool", "Agent", {"prompt": str}),

    # ── Bash (10) ──
    ToolTest("Run ls -la in the current directory", "Bash", {"command": str}),
    ToolTest("Execute 'cargo test' to run the test suite", "Bash", {"command": str}),
    ToolTest("Run 'git log --oneline -10' to see recent commits", "Bash", {"command": str}),
    ToolTest("Check disk usage with df -h", "Bash", {"command": str}),
    ToolTest("Run 'npm install' in the background", "Bash", {"command": str}, {"run_in_background": True}),
    ToolTest("Execute 'docker ps' to see running containers", "Bash", {"command": str}),
    ToolTest("Run the build with a 5 minute timeout: make -j$(nproc)", "Bash", {"command": str}, {"timeout": 300000}),
    ToolTest("Check the python version with python3 --version", "Bash", {"command": str}),
    ToolTest("Run 'find . -name *.rs -type f | wc -l' to count Rust files", "Bash", {"command": str}),
    ToolTest("Execute 'cat /proc/cpuinfo | head -20'", "Bash", {"command": str}),

    # ── Read (10) ──
    ToolTest("Read the file /home/user/project/src/main.rs", "Read", {"file_path": str}),
    ToolTest("Show me /etc/hosts", "Read", {"file_path": str}),
    ToolTest("Read /app/config.toml starting from line 50", "Read", {"file_path": str}, {"offset": 50}),
    ToolTest("Read the first 100 lines of /home/user/log.txt", "Read", {"file_path": str}, {"limit": 100}),
    ToolTest("Show me lines 200-400 of /workspace/src/lib.rs", "Read", {"file_path": str}, {"offset": 200, "limit": 200}),
    ToolTest("Read /home/user/report.pdf pages 1 through 5", "Read", {"file_path": str}, {"pages": "1-5"}),
    ToolTest("Open /workspace/Cargo.toml", "Read", {"file_path": str}),
    ToolTest("Show me /app/src/database.py from line 300, 200 lines", "Read", {"file_path": str}, {"offset": 300, "limit": 200}),
    ToolTest("Read the file at /tmp/output.log", "Read", {"file_path": str}),
    ToolTest("Show /workspace/README.md", "Read", {"file_path": str}),

    # ── Edit (10) ──
    ToolTest("In /app/src/main.rs, replace 'fn old_name' with 'fn new_name'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("Edit /workspace/config.toml: change 'port = 8080' to 'port = 9090'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("In /app/index.js, replace all occurrences of 'var' with 'const'", "Edit", {"file_path": str, "old_string": str, "new_string": str}, {"replace_all": True}),
    ToolTest("Fix the typo in /home/user/doc.md: change 'teh' to 'the'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("In /app/lib.rs change 'use std::io;' to 'use std::io::prelude::*;'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("Replace 'TODO: implement' with 'return Ok(())' in /workspace/handler.rs", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("Edit /app/style.css: change 'color: red' to 'color: blue'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("In /workspace/test.py, replace 'assertEqual' with 'assert_equal' everywhere", "Edit", {"file_path": str, "old_string": str, "new_string": str}, {"replace_all": True}),
    ToolTest("Change the database URL in /app/.env from 'localhost' to '10.0.0.5'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),
    ToolTest("In /workspace/main.go change 'log.Println' to 'log.Printf'", "Edit", {"file_path": str, "old_string": str, "new_string": str}),

    # ── Write (10) ──
    ToolTest("Create a new file /workspace/hello.py with a hello world script", "Write", {"file_path": str, "content": str}),
    ToolTest("Write a Dockerfile to /app/Dockerfile with a python 3.13 base image", "Write", {"file_path": str, "content": str}),
    ToolTest("Create /workspace/.gitignore with entries for node_modules and .env", "Write", {"file_path": str, "content": str}),
    ToolTest("Write a basic Cargo.toml to /workspace/Cargo.toml for a project named 'myapp'", "Write", {"file_path": str, "content": str}),
    ToolTest("Create /tmp/test.json with {\"key\": \"value\"}", "Write", {"file_path": str, "content": str}),
    ToolTest("Write an empty __init__.py to /app/src/__init__.py", "Write", {"file_path": str, "content": str}),
    ToolTest("Create a Makefile at /workspace/Makefile with a build target", "Write", {"file_path": str, "content": str}),
    ToolTest("Write a shell script to /workspace/run.sh that starts the server", "Write", {"file_path": str, "content": str}),
    ToolTest("Create /workspace/config.yaml with host: 0.0.0.0 and port: 8080", "Write", {"file_path": str, "content": str}),
    ToolTest("Write a basic index.html to /workspace/public/index.html", "Write", {"file_path": str, "content": str}),

    # ── Glob (10) ──
    ToolTest("Find all Rust files in the project", "Glob", {"pattern": str}),
    ToolTest("List all .toml files under src/", "Glob", {"pattern": str}),
    ToolTest("Find all test files matching *_test.py", "Glob", {"pattern": str}),
    ToolTest("Glob for **/*.tsx in the components directory", "Glob", {"pattern": str}, {"path": str}),
    ToolTest("Find all markdown files", "Glob", {"pattern": str}),
    ToolTest("List all .json config files in configs/", "Glob", {"pattern": str}),
    ToolTest("Find all Dockerfiles in the repo", "Glob", {"pattern": str}),
    ToolTest("Glob for *.cpp and *.h files", "Glob", {"pattern": str}),
    ToolTest("Find all migration SQL files", "Glob", {"pattern": str}),
    ToolTest("List .env* files in the project root", "Glob", {"pattern": str}),

    # ── Grep (10) ──
    ToolTest("Search for 'TODO' in all files", "Grep", {"pattern": str}),
    ToolTest("Find all occurrences of 'unsafe' in Rust files", "Grep", {"pattern": str}, {"type": "rust"}),
    ToolTest("Grep for 'console.log' in JavaScript files, case insensitive", "Grep", {"pattern": str}, {"-i": True}),
    ToolTest("Search for 'fn main' in src/ showing file content", "Grep", {"pattern": str}, {"output_mode": "content"}),
    ToolTest("Count how many times 'import' appears in Python files", "Grep", {"pattern": str}, {"output_mode": "count", "type": "py"}),
    ToolTest("Find 'password' in all files with 3 lines of context", "Grep", {"pattern": str}, {"-C": 3}),
    ToolTest("Search for the pattern 'async fn \\w+' in .rs files", "Grep", {"pattern": str}, {"glob": "*.rs"}),
    ToolTest("Grep for 'SELECT.*FROM' in SQL files, multiline mode", "Grep", {"pattern": str}, {"multiline": True}),
    ToolTest("Find 'error' in log files, show 2 lines after each match", "Grep", {"pattern": str}, {"-A": 2}),
    ToolTest("Search for 'deprecated' in the codebase, limit to first 10 results", "Grep", {"pattern": str}, {"head_limit": 10}),

    # ── PresentFile (10) ──
    ToolTest("Send the user the report at /workspace/outputs/report.pdf", "PresentFile", {"file_path": str}),
    ToolTest("Present /workspace/outputs/chart.png to the user", "PresentFile", {"file_path": str}),
    ToolTest("Give the user the file /workspace/outputs/data.csv", "PresentFile", {"file_path": str}),
    ToolTest("Share /workspace/outputs/analysis.xlsx with the user", "PresentFile", {"file_path": str}),
    ToolTest("Present the generated diagram at /workspace/outputs/arch.svg", "PresentFile", {"file_path": str}),
    ToolTest("Send the user /workspace/outputs/logs.zip", "PresentFile", {"file_path": str}),
    ToolTest("Deliver /workspace/outputs/summary.md to the user", "PresentFile", {"file_path": str}),
    ToolTest("Present /workspace/outputs/benchmark_results.json", "PresentFile", {"file_path": str}),
    ToolTest("Give the user the compiled binary at /workspace/outputs/app", "PresentFile", {"file_path": str}),
    ToolTest("Share the exported config at /workspace/outputs/config.toml", "PresentFile", {"file_path": str}),

    # ── WebFetch (10) ──
    ToolTest("Fetch https://docs.rs/serde/latest and extract the derive macro docs", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Get https://react.dev/learn and summarize the quick start guide", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Fetch https://arxiv.org/abs/2301.00001 and extract the abstract", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Read https://github.com/tokio-rs/tokio and get the project description", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Fetch https://blog.rust-lang.org and list the latest blog post titles", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Get the pricing info from https://example.com/pricing", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Fetch https://docs.docker.com/compose and find the volume mount syntax", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Read https://crates.io/crates/axum and get the version number", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Fetch https://huggingface.co/models and find trending models", "WebFetch", {"url": str, "prompt": str}),
    ToolTest("Get https://en.wikipedia.org/wiki/Transformer and extract the architecture description", "WebFetch", {"url": str, "prompt": str}),

    # ── NotebookEdit (10) ──
    ToolTest("Replace cell 0 in /workspace/analysis.ipynb with 'import pandas as pd'", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}),
    ToolTest("Insert a new cell at position 3 in /workspace/notebook.ipynb with a matplotlib plot", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}, {"edit_mode": "insert"}),
    ToolTest("Delete cell 5 from /workspace/experiment.ipynb", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}, {"edit_mode": "delete"}),
    ToolTest("Update cell 2 in /app/demo.ipynb to print('hello world')", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}),
    ToolTest("Replace cell 10 in /workspace/train.ipynb with the model training loop", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}),
    ToolTest("Insert a markdown cell at position 0 in /workspace/report.ipynb with '# Analysis Report'", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}, {"edit_mode": "insert"}),
    ToolTest("Delete cell 7 from /workspace/scratch.ipynb", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}, {"edit_mode": "delete"}),
    ToolTest("Update cell 1 in /workspace/data.ipynb with data loading code", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}),
    ToolTest("Replace cell 4 in /workspace/viz.ipynb with a seaborn heatmap", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}),
    ToolTest("Insert a code cell at position 6 in /workspace/test.ipynb", "NotebookEdit", {"notebook_path": str, "cell_number": int, "new_source": str}, {"edit_mode": "insert"}),

    # ── AskUserQuestion (10) ──
    ToolTest("Ask the user whether they want to use SQLite or PostgreSQL for the database", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Ask the user to choose between REST API or GraphQL", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Present the user with a choice: deploy to AWS, GCP, or Azure", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Ask if they want dark mode or light mode for the UI theme", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Let the user pick the testing framework: pytest, unittest, or nose2", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Ask the user which package manager to use: npm, yarn, or pnpm", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Ask the user to select which features to enable: auth, logging, caching, metrics (allow multiple)", "AskUserQuestion", {"question": str, "options": list}, {"multiSelect": True}),
    ToolTest("Ask whether to use TypeScript strict mode or not", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Let the user choose the log level: debug, info, warn, or error", "AskUserQuestion", {"question": str, "options": list}),
    ToolTest("Ask if they prefer monorepo or polyrepo structure", "AskUserQuestion", {"question": str, "options": list}),

    # ── LSP (10) ──
    ToolTest("Go to the definition of the function at line 42, column 10 in /app/src/main.rs", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Find all references to the symbol at line 15 col 5 in /workspace/lib.py", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Show hover info for the type at line 100, character 20 in /app/types.ts", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("List all symbols in /workspace/src/handlers.rs", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Search the workspace for the symbol 'DatabasePool'", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Go to the implementation of the trait at line 55 col 8 in /app/src/traits.rs", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Find what calls the function at line 200, column 12 in /workspace/api.py", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Get the definition of 'handle_request' at line 30 col 15 in /app/server.rs", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Show all outgoing calls from the function at line 75 col 4 in /workspace/core.py", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),
    ToolTest("Find references to the variable at line 10, character 8 in /app/config.ts", "LSP", {"operation": str, "file_path": str, "line": int, "character": int}),

    # ── Skill (10) ──
    ToolTest("Run the commit skill", "Skill", {"skill": str}),
    ToolTest("Execute the code-review skill on the current diff", "Skill", {"skill": str}),
    ToolTest("Run the init skill to create a CLAUDE.md", "Skill", {"skill": str}),
    ToolTest("Use the security-review skill", "Skill", {"skill": str}),
    ToolTest("Run /review on PR 42", "Skill", {"skill": str}, {"args": str}),
    ToolTest("Execute the simplify skill", "Skill", {"skill": str}),
    ToolTest("Run the verify skill to test the changes", "Skill", {"skill": str}),
    ToolTest("Use the run skill to start the app", "Skill", {"skill": str}),
    ToolTest("Execute /loop with 5m interval", "Skill", {"skill": str}, {"args": str}),
    ToolTest("Run the schedule skill to set up a daily check", "Skill", {"skill": str}),

    # ── SearchConversation (10) ──
    ToolTest("Search our conversation for when I mentioned 'database migration'", "SearchConversation", {"pattern": str}),
    ToolTest("Find where I talked about the API endpoint earlier", "SearchConversation", {"pattern": str}),
    ToolTest("Search my messages for 'config' case insensitively", "SearchConversation", {"pattern": str}, {"-i": True}),
    ToolTest("Look through our chat for 'deploy' with 2 lines of context", "SearchConversation", {"pattern": str}, {"context": 2}),
    ToolTest("Find the last time I mentioned the test failures", "SearchConversation", {"pattern": str}),
    ToolTest("Search for 'port.*8080' in our conversation", "SearchConversation", {"pattern": str}),
    ToolTest("Did I mention anything about Docker earlier? Search for it", "SearchConversation", {"pattern": str}),
    ToolTest("Find where I specified the file path earlier, limit to 5 results", "SearchConversation", {"pattern": str}, {"limit": 5}),
    ToolTest("Search for when I asked about the Rust compiler error", "SearchConversation", {"pattern": str}),
    ToolTest("Look for any mention of 'timeout' in my previous messages", "SearchConversation", {"pattern": str}),

    # ── SendMessage (10) ──
    ToolTest("Send a message to Alice saying the build is complete", "SendMessage", {"to": str, "message": str}),
    ToolTest("Tell Bob that the PR is ready for review", "SendMessage", {"to": str, "message": str}),
    ToolTest("Message the team lead: 'deployment finished successfully'", "SendMessage", {"to": str, "message": str}),
    ToolTest("Send 'tests passing' to the CI agent", "SendMessage", {"to": str, "message": str}),
    ToolTest("Notify Carol that the database migration is done", "SendMessage", {"to": str, "message": str}),
    ToolTest("Broadcast to all: 'system maintenance at 2am'", "SendMessage", {"to": str, "message": str}),
    ToolTest("Message Dave with a summary: 'Fixed the auth bug in login handler'", "SendMessage", {"to": str, "message": str, "summary": str}),
    ToolTest("Tell the review agent that changes are pushed", "SendMessage", {"to": str, "message": str}),
    ToolTest("Send Eve the error log excerpt", "SendMessage", {"to": str, "message": str}),
    ToolTest("Message Frank: 'the staging environment is ready'", "SendMessage", {"to": str, "message": str}),

    # ── SendUserMessage (10) ──
    ToolTest("Tell the user that the refactoring is complete", "SendUserMessage", {"message": str}),
    ToolTest("Send the user a proactive status update about the build", "SendUserMessage", {"message": str}, {"status": "proactive"}),
    ToolTest("Notify the user with the test results summary", "SendUserMessage", {"message": str}),
    ToolTest("Send the user a message with the diff attached from /tmp/changes.diff", "SendUserMessage", {"message": str}, {"attachments": list}),
    ToolTest("Tell the user the deployment succeeded", "SendUserMessage", {"message": str}),
    ToolTest("Proactively inform the user about the completed analysis", "SendUserMessage", {"message": str}, {"status": "proactive"}),
    ToolTest("Send the user the benchmark results", "SendUserMessage", {"message": str}),
    ToolTest("Notify the user that the cache has been cleared", "SendUserMessage", {"message": str}),
    ToolTest("Send the user a message with the log file attached: /tmp/app.log", "SendUserMessage", {"message": str}, {"attachments": list}),
    ToolTest("Tell the user all migrations have been applied", "SendUserMessage", {"message": str}),
]


# ── Scoring ──────────────────────────────────────────────────────────

@dataclass
class ToolCallScore:
    test: ToolTest
    raw_response: str
    tool_call_json: dict | None = None
    valid_json: bool = False
    correct_tool: bool = False
    required_params_present: bool = False
    param_types_valid: bool = False
    enum_values_valid: bool = False
    constraints_respected: bool = False
    no_hallucinated_params: bool = False
    overall_pass: bool = False
    error: str | None = None


def _check_type(value, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type in ("integer", "number"):
        return isinstance(value, (int, float))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def score_tool_call(test: ToolTest, raw_response: str, tool_call: dict | None) -> ToolCallScore:
    sc = ToolCallScore(test=test, raw_response=raw_response)

    if tool_call is None:
        sc.error = "no tool call in response"
        return sc

    sc.tool_call_json = tool_call
    sc.valid_json = True

    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    sc.correct_tool = name == test.expected_tool

    if not sc.correct_tool:
        sc.error = f"wrong tool: got {name!r}, expected {test.expected_tool!r}"
        return sc

    args_str = fn.get("arguments", "{}")
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        sc.error = f"invalid JSON in arguments: {args_str[:200]}"
        return sc

    schema = _get_schema(name)
    if schema is None:
        sc.error = f"no schema for tool {name!r}"
        return sc

    props = schema.get("properties", {})
    required = schema.get("required", [])

    sc.required_params_present = all(r in args for r in required)

    types_ok = True
    for k, v in args.items():
        if k in props:
            expected = props[k].get("type", "string")
            if not _check_type(v, expected):
                types_ok = False
                break
    sc.param_types_valid = types_ok

    enums_ok = True
    for k, v in args.items():
        if k in props and "enum" in props[k]:
            if v not in props[k]["enum"]:
                enums_ok = False
                break
    sc.enum_values_valid = enums_ok

    constraints_ok = True
    for k, v in args.items():
        if k in props and isinstance(v, (int, float)):
            if "minimum" in props[k] and v < props[k]["minimum"]:
                constraints_ok = False
            if "maximum" in props[k] and v > props[k]["maximum"]:
                constraints_ok = False
    sc.constraints_respected = constraints_ok

    sc.no_hallucinated_params = all(k in props for k in args)

    sc.overall_pass = all([
        sc.valid_json,
        sc.correct_tool,
        sc.required_params_present,
        sc.param_types_valid,
        sc.enum_values_valid,
        sc.constraints_respected,
        sc.no_hallucinated_params,
    ])

    return sc


# ── API call with tools ─────────────────────────────────────────────

def chat_complete_tools(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = 120.0,
    api_key: str = "not-needed",
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


# ── Runner ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an AI assistant with access to various tools. "
    "When the user asks you to do something, use the appropriate tool. "
    "Always use a tool call to fulfill the request — do not respond with plain text alone. "
    "Pick the single most appropriate tool for each request."
)


def run_toolcall_benchmark(
    base_url: str,
    model: str,
    dump_path: Path | None = None,
    api_key: str = "not-needed",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = 120.0,
) -> list[ToolCallScore]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    scores: list[ToolCallScore] = []

    total = len(TOOL_TESTS)
    print(f"Tool calling benchmark: {total} tests across {len(TOOL_DEFINITIONS)} tools\n", flush=True)

    for i, test in enumerate(TOOL_TESTS, 1):
        messages.append({"role": "user", "content": test.prompt})
        print(f"[{i}/{total}] {test.expected_tool:<25} {test.prompt[:60]}...", end="", flush=True)

        start = time.monotonic()
        try:
            resp = chat_complete_tools(
                base_url=base_url,
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
            )
            latency = time.monotonic() - start
            raw = json.dumps(resp)

            choice = resp.get("choices", [{}])[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls", [])
            tool_call = tool_calls[0] if tool_calls else None

            sc = score_tool_call(test, raw, tool_call)
            mark = "✓" if sc.overall_pass else "✗"
            err_note = f"  ({sc.error})" if sc.error else ""
            print(f"  {mark} {latency:.1f}s{err_note}", flush=True)

            if tool_call:
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": tool_calls,
                })
                mock = MOCK_RESPONSES.get(tool_call["function"]["name"], '{"status": "ok"}')
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{i}"),
                    "content": mock,
                })
            else:
                messages.append({"role": "assistant", "content": msg.get("content", "")})

        except Exception as e:
            latency = time.monotonic() - start
            sc = ToolCallScore(test=test, raw_response="", error=str(e))
            print(f"  ERROR {latency:.1f}s  {e}", flush=True)
            messages.append({"role": "assistant", "content": f"Error: {e}"})

        scores.append(sc)

    _print_summary(scores)

    if dump_path:
        _dump_results(scores, model, base_url, dump_path)

    return scores


def _print_summary(scores: list[ToolCallScore]) -> None:
    total = len(scores)
    passed = sum(1 for s in scores if s.overall_pass)
    valid_json = sum(1 for s in scores if s.valid_json)
    correct_tool = sum(1 for s in scores if s.correct_tool)
    req_params = sum(1 for s in scores if s.required_params_present)
    types_ok = sum(1 for s in scores if s.param_types_valid)
    enums_ok = sum(1 for s in scores if s.enum_values_valid)
    constraints_ok = sum(1 for s in scores if s.constraints_respected)
    no_halluc = sum(1 for s in scores if s.no_hallucinated_params)

    print(f"\n{'='*60}", flush=True)
    print(f"  TOOL CALLING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Overall pass:          {passed}/{total}", flush=True)
    print(f"  Valid JSON:            {valid_json}/{total}", flush=True)
    print(f"  Correct tool:          {correct_tool}/{total}", flush=True)
    print(f"  Required params:       {req_params}/{total}", flush=True)
    print(f"  Param types valid:     {types_ok}/{total}", flush=True)
    print(f"  Enum values valid:     {enums_ok}/{total}", flush=True)
    print(f"  Constraints respected: {constraints_ok}/{total}", flush=True)
    print(f"  No hallucinated params:{no_halluc}/{total}", flush=True)

    tools_seen: dict[str, list[ToolCallScore]] = {}
    for s in scores:
        name = s.test.expected_tool
        tools_seen.setdefault(name, []).append(s)

    print(f"\n  per-tool:", flush=True)
    for name, tool_scores in tools_seen.items():
        p = sum(1 for s in tool_scores if s.overall_pass)
        t = len(tool_scores)
        mark = "✓" if p == t else ("~" if p > 0 else "✗")
        print(f"    {mark} {name:<25} {p}/{t}", flush=True)


def _dump_results(scores: list[ToolCallScore], model: str, base_url: str, dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for sc in scores:
        results.append({
            "prompt": sc.test.prompt,
            "expected_tool": sc.test.expected_tool,
            "overall_pass": sc.overall_pass,
            "valid_json": sc.valid_json,
            "correct_tool": sc.correct_tool,
            "required_params_present": sc.required_params_present,
            "param_types_valid": sc.param_types_valid,
            "enum_values_valid": sc.enum_values_valid,
            "constraints_respected": sc.constraints_respected,
            "no_hallucinated_params": sc.no_hallucinated_params,
            "error": sc.error,
            "tool_call": sc.tool_call_json,
        })
    payload = {
        "benchmark": "toolcall",
        "model": model,
        "base_url": base_url,
        "total_tests": len(scores),
        "total_passed": sum(1 for s in scores if s.overall_pass),
        "results": results,
    }
    dump_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults dumped to {dump_path}", flush=True)
