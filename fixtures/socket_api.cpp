#include "socket_api.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>

bool SocketApi::init(const char* path) {
    signal(SIGPIPE, SIG_IGN);

    strncpy(sock_path_, path, sizeof(sock_path_) - 1);
    sock_path_[sizeof(sock_path_) - 1] = '\0';

    unlink(sock_path_);

    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path_, sizeof(addr.sun_path) - 1);

    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    chmod(sock_path_, 0666);

    if (listen(listen_fd_, 4) < 0 || fcntl(listen_fd_, F_SETFL, O_NONBLOCK) < 0) {
        close(listen_fd_);
        unlink(sock_path_);
        listen_fd_ = -1;
        return false;
    }

    json_len_ = 0;
    return true;
}

void SocketApi::shutdown() {
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        unlink(sock_path_);
        listen_fd_ = -1;
    }
    delete[] json_buf_;
    json_buf_ = nullptr;
    json_cap_ = 0;
    json_len_ = 0;
}

// Safe snprintf helper: advances off, clamps to cap to prevent overflow
static inline void safe_append(char* buf, int cap, int& off, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));

static inline void safe_append(char* buf, int cap, int& off, const char* fmt, ...) {
    if (off >= cap - 1) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + off, cap - off, fmt, ap);
    va_end(ap);
    if (n > 0) {
        off += n;
        if (off >= cap) off = cap - 1;
    }
}

static inline void safe_char(char* buf, int cap, int& off, char ch) {
    if (off < cap - 1) buf[off++] = ch;
}

int SocketApi::build_json(char* buf, int cap,
                          const GpuMonitor& gpu, const CpuMonitor& cpu,
                          const MemoryMonitor& mem, const NetworkMonitor& net,
                          const PowerMonitor& power) {
    int off = 0;
    safe_append(buf, cap, off, "{\"timestamp\":%ld", (long)time(nullptr));

    // GPU
    safe_append(buf, cap, off, ",\"gpu\":[");
    auto& gs = gpu.stats();
    for (size_t i = 0; i < gs.size(); i++) {
        auto& g = gs[i];
        if (i > 0) safe_char(buf, cap, off, ',');
        safe_append(buf, cap, off,
            "{\"index\":%d,\"usage\":%d,\"mem_used_mb\":%d,\"mem_total_mb\":%d,"
            "\"mem_percent\":%d,\"temp_edge\":%d,\"temp_junction\":%d,\"temp_mem\":%d,"
            "\"sclk_mhz\":%d,\"mclk_mhz\":%d,\"power_avg_w\":%d,\"power_cap_w\":%d,"
            "\"fan_rpm\":%d,\"fan_percent\":%d,\"name\":\"%.63s\",\"power_level\":\"%.31s\"}",
            g.index, g.usage, g.mem_used_mb, g.mem_total_mb,
            g.mem_percent, g.temp_edge, g.temp_junction, g.temp_mem,
            g.sclk_mhz, g.mclk_mhz, g.power_avg_w, g.power_cap_w,
            g.fan_rpm, g.fan_percent, g.name.c_str(), g.power_level.c_str());
    }
    safe_append(buf, cap, off, "]");

    // CPU summary
    auto& cs = cpu.summary();
    safe_append(buf, cap, off,
        ",\"cpu\":{\"avg_busy\":%d,\"avg_mhz\":%d,\"pkg_temp\":%d,"
        "\"ccd1_temp\":%d,\"ccd2_temp\":%d,\"core_count\":%d,\"thread_count\":%d,\"cores\":[",
        cs.avg_busy, cs.avg_mhz, cs.pkg_temp, cs.ccd1_temp, cs.ccd2_temp,
        cpu.core_count(), cpu.thread_count());

    auto& cores = cpu.cores();
    for (size_t i = 0; i < cores.size(); i++) {
        auto& c = cores[i];
        if (i > 0) safe_char(buf, cap, off, ',');
        safe_append(buf, cap, off,
            "{\"core_id\":%d,\"busy_percent\":%d,\"freq_mhz\":%d}",
            c.core_id, c.busy_percent, c.freq_mhz);
    }
    safe_append(buf, cap, off, "]}");

    // Memory
    auto& ms = mem.stats();
    safe_append(buf, cap, off,
        ",\"memory\":{\"total_kb\":%ld,\"used_kb\":%ld,\"avail_kb\":%ld,"
        "\"cached_kb\":%ld,\"percent\":%d}",
        (long)ms.total_kb, (long)ms.used_kb, (long)ms.avail_kb,
        (long)ms.cached_kb, ms.percent);

    // Network
    safe_append(buf, cap, off, ",\"network\":[");
    auto& ns = net.stats();
    for (size_t i = 0; i < ns.size(); i++) {
        auto& n = ns[i];
        if (i > 0) safe_char(buf, cap, off, ',');
        safe_append(buf, cap, off,
            "{\"name\":\"%.31s\",\"link_speed_mbps\":%d,\"rx_bps\":%ld,\"tx_bps\":%ld,"
            "\"rx_pps\":%ld,\"tx_pps\":%ld,\"rx_pct\":%d,\"tx_pct\":%d,"
            "\"rx_max_mbps\":%d,\"rx_min_mbps\":%d,\"tx_max_mbps\":%d,\"tx_min_mbps\":%d}",
            n.name.c_str(), n.link_speed_mbps,
            (long)n.rx_bps, (long)n.tx_bps, (long)n.rx_pps, (long)n.tx_pps,
            n.rx_pct, n.tx_pct,
            n.rx_max_mbps, n.rx_min_mbps, n.tx_max_mbps, n.tx_min_mbps);
    }
    safe_append(buf, cap, off, "]");

    // Power
    auto& ps = power.stats();
    safe_append(buf, cap, off,
        ",\"power\":{\"available\":%s,\"pkg_watt\":%.1f}",
        ps.available ? "true" : "false", ps.pkg_watt);

    safe_append(buf, cap, off, "}\n");
    buf[off] = '\0';
    return off;
}

void SocketApi::update_json(const GpuMonitor& gpu,
                            const CpuMonitor& cpu,
                            const MemoryMonitor& mem,
                            const NetworkMonitor& net,
                            const PowerMonitor& power) {
    if (!json_buf_) {
        // First call: measure with temporary buffer, then allocate 2.25x
        char tmp[65536];
        int len = build_json(tmp, static_cast<int>(sizeof(tmp)), gpu, cpu, mem, net, power);
        json_cap_ = static_cast<int>(len * 2.25);
        if (json_cap_ < 256) json_cap_ = 256;
        json_buf_ = new char[json_cap_];
        memcpy(json_buf_, tmp, len + 1);
        json_len_ = len;
        return;
    }

    int off = build_json(json_buf_, json_cap_, gpu, cpu, mem, net, power);
    if (off >= json_cap_ - 1) {
        // Truncated — write error into buffer
        json_len_ = snprintf(json_buf_, json_cap_,
                             "{\"error\":\"json exceeds buffer capacity\"}\n");
        return;
    }
    json_len_ = off;
}

void SocketApi::poll_clients() {
    if (listen_fd_ < 0 || json_len_ <= 0) return;
    int client = accept(listen_fd_, nullptr, nullptr);
    if (client < 0) return;
    ssize_t ret __attribute__((unused)) = write(client, json_buf_, json_len_);
    close(client);
}
