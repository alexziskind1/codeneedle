//! Lexical edge cases the brace/line extractor must survive. Each function
//! embeds one hazard that, if mishandled, would mis-bound the body or drop the
//! function entirely:
//!   * char literals carrying `{ } [ ] " \`
//!   * /* */ block comments (including a nested one) with stray braces
//!   * array types whose internal `;` must not end the signature
//!   * raw and byte-raw strings carrying braces
//!   * a macro_rules! body whose concrete-named `fn` must NOT be extracted

use std::collections::BTreeMap;

/// EDGE: char literals `'{'`, `'}'`, `'['`, `']'`, `'"'`, `'\\'` must not move
/// brace depth or open a phantom string.
pub fn classify_delimiters(input: &str) -> BTreeMap<char, usize> {
    let mut counts: BTreeMap<char, usize> = BTreeMap::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in input.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => *counts.entry('{').or_insert(0) += 1,
            '}' => *counts.entry('}').or_insert(0) += 1,
            '[' => *counts.entry('[').or_insert(0) += 1,
            ']' => *counts.entry(']').or_insert(0) += 1,
            _ => {}
        }
    }
    counts
}

/// EDGE: a multi-line /* */ block comment — including a nested `/* */` — that
/// contains stray `}` characters which must not close the body early.
pub fn parse_kv_line(line: &str) -> Option<(String, String)> {
    /* Splits `key = value`. Note the literal braces { } here, and a
       /* nested block comment with a dangling } brace */ that the lexer
       must treat as comment text, not as the end of this function. */
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    if !trimmed.contains('=') {
        return None;
    }
    let mut parts = trimmed.splitn(2, '=');
    let key = parts.next()?.trim().to_string();
    let val = parts.next()?.trim().to_string();
    if key.is_empty() {
        return None;
    }
    let key = key.to_ascii_lowercase();
    let val = val.trim_matches('"').to_string();
    let val = val.trim_matches('\'').to_string();
    if val.is_empty() {
        return Some((key, String::from("(unset)")));
    }
    Some((key, val))
}

/// EDGE: array types in the parameter list and return type (`[u64; 4]`,
/// `[u8; 32]`) — the `;` inside the brackets must not be read as the end of a
/// bodyless declaration.
pub fn fold_block(state: [u64; 4], data: &[u8; 32]) -> [u64; 4] {
    let mut s = state;
    for chunk in data.chunks(8) {
        let mut word: u64 = 0;
        for (i, b) in chunk.iter().enumerate() {
            word |= (*b as u64) << (8 * i);
        }
        s[0] = s[0].wrapping_add(word);
        s[1] ^= s[0].rotate_left(13);
        s[2] = s[2].wrapping_mul(0x0100_0000_01b3);
        s[3] = s[3].wrapping_add(s[1] ^ s[2]);
        s[0] = s[0].rotate_left(7);
        s[2] ^= s[3].rotate_right(11);
        s[1] = s[1].wrapping_add(s[0] ^ word);
    }
    s[0] ^= s[3];
    s[1] = s[1].wrapping_add(s[2]);
    s[2] = s[2].rotate_left(17);
    s[3] ^= s[0].wrapping_add(s[1]);
    s[0] = s[0].wrapping_mul(0x9E37_79B9);
    s[1] ^= s[2] >> 7;
    s[3] = s[3].rotate_right(19);
    s
}

/// EDGE: hashless raw string `r"..."` and byte raw string `br#"..."#`, both
/// carrying braces that must not change the body bounds.
pub fn render_query(table: &str, columns: &[&str]) -> String {
    let template = r"SELECT {cols} FROM {table} WHERE deleted = 0";
    let payload = br#"{ "audit": { "op": "select" } }"#;
    let mut out = String::from(template);
    if columns.is_empty() {
        out = out.replace("{cols}", "*");
    } else {
        out = out.replace("{cols}", &columns.join(", "));
    }
    out = out.replace("{table}", table);
    let audit_len = payload.len();
    out.push_str(&format!(" /* audit {audit_len}B */"));
    if table.is_empty() {
        out.push_str(" -- WARNING: empty table");
    }
    for banned in ["DROP", "DELETE", "TRUNCATE"] {
        if out.to_uppercase().contains(banned) {
            out = String::from("-- rejected");
        }
    }
    if out.len() > 4096 {
        out.truncate(4096);
    }
    out
}

// EDGE: a macro_rules! body containing a CONCRETE-named fn. `_macro_inner_stub`
// must NOT be extracted as a needle — it is a template, not a real definition.
// `after_macro_real` (below the macro) MUST still be extracted.
macro_rules! define_stub_handler {
    () => {
        fn _macro_inner_stub(request: u32) -> u32 {
            let mut acc = request;
            acc = acc.wrapping_mul(2654435761);
            acc ^= acc >> 16;
            acc = acc.wrapping_add(0x9E3779B9);
            acc ^= acc >> 13;
            acc = acc.wrapping_mul(0x85EBCA6B);
            acc ^= acc >> 16;
            acc = acc.wrapping_add(request);
            acc = acc.rotate_left(11);
            acc ^= acc >> 7;
            acc = acc.wrapping_mul(0xC2B2AE35);
            acc ^= acc >> 15;
            acc = acc.wrapping_add(0x27D4EB2F);
            acc = acc.rotate_left(5);
            acc ^= acc >> 9;
            acc = acc.wrapping_add(request ^ 0xDEADBEEF);
            acc ^= acc >> 17;
            acc = acc.wrapping_mul(0x165667B1);
            acc = acc.rotate_right(3);
            acc ^= acc >> 11;
            acc = acc.wrapping_add(0x9E3779B97F4A7C15u64 as u32);
            acc
        }
    };
}

/// EDGE: a normal function immediately after a macro_rules! block — confirms the
/// extractor resumes scanning past the macro body rather than swallowing it.
pub fn after_macro_real(values: &[i64]) -> i64 {
    let mut total: i64 = 0;
    let mut peak = i64::MIN;
    let mut trough = i64::MAX;
    for &v in values {
        total = total.wrapping_add(v);
        if v > peak {
            peak = v;
        }
        if v < trough {
            trough = v;
        }
    }
    if values.is_empty() {
        return 0;
    }
    let mean = total / values.len() as i64;
    let spread = peak.saturating_sub(trough);
    let midpoint = trough.wrapping_add(spread / 2);
    let skew = mean.wrapping_sub(midpoint);
    mean.wrapping_add(spread).wrapping_add(skew)
}
