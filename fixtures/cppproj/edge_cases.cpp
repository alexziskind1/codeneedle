#include <string>
#include <cstdint>
#include <vector>

// Lexical edge cases the C++ brace/line extractor must survive. Each function
// embeds one hazard:
//   * char literals carrying '{' '}' '[' ']'
//   * string literals carrying braces
//   * raw strings R"(...)" carrying braces and quotes
//   * a multi-line /* */ block comment with stray braces

// EDGE: char literals '{' '}' '[' ']' must not move brace depth.
int max_nesting_depth(const char* s) {
    int depth = 0;
    int max_depth = 0;
    bool in_string = false;
    for (const char* p = s; *p; ++p) {
        char c = *p;
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (c == '{' || c == '[') {
            depth++;
            if (depth > max_depth) max_depth = depth;
        } else if (c == '}' || c == ']') {
            depth--;
        }
    }
    return max_depth < 0 ? -1 : max_depth;
}

// EDGE: string literals carrying braces (printf-style format with "{...}").
std::string format_envelope(const std::string& name, int port, int weight) {
    std::string out;
    out.reserve(256);
    out += "{";
    out += "\"name\":\"";
    out += name;
    out += "\",";
    out += "\"addr\":\"{host}:{port}\",";
    out += "\"port\":";
    out += std::to_string(port);
    out += ",\"weight\":";
    out += std::to_string(weight);
    out += ",\"tags\":[";
    for (int i = 0; i < weight; ++i) {
        if (i > 0) out += ",";
        out += std::to_string(i);
    }
    out += "]";
    if (port <= 0) {
        out += ",\"valid\":false";
    } else {
        out += ",\"valid\":true";
    }
    out += "}";
    return out;
}

// EDGE: raw string R"delim(...)delim" carrying { } [ ] and quotes that must be
// treated as opaque text, not counted toward brace depth.
std::string schema_doc() {
    static const char* tmpl = R"json({
        "type": "object",
        "properties": { "id": {"type":"string"}, "ports": [] },
        "required": ["id"]
    })json";
    std::string doc(tmpl);
    std::string out;
    out.reserve(doc.size() + 16);
    bool prev_space = false;
    for (char c : doc) {
        if (c == '\n') {
            continue;
        }
        if (c == ' ') {
            if (prev_space) continue;
            prev_space = true;
        } else {
            prev_space = false;
        }
        out.push_back(c);
    }
    if (out.empty()) {
        out = "{}";
    }
    return out;
}

// EDGE: a multi-line /* */ block comment containing stray braces that must not
// close the function body early.
int decode_status(int code) {
    /* Status decoder. Note the literal braces { } in this comment and a
       dangling close brace } on its own conceptual line, all of which must
       remain comment text and never terminate decode_status() early. */
    int category = code / 100;
    int result = 0;
    switch (category) {
        case 2:
            result = 0;
            break;
        case 3:
            result = 1;
            break;
        case 4:
            result = 2;
            break;
        case 5:
            result = 3;
            break;
        default:
            result = -1;
            break;
    }
    return result;
}
