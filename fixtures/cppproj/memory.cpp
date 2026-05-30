#include "memory.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

MemoryMonitor::~MemoryMonitor() {
    if (meminfo_fd_ >= 0) close(meminfo_fd_);
}

void MemoryMonitor::poll() {
    if (meminfo_fd_ < 0) {
        meminfo_fd_ = open("/proc/meminfo", O_RDONLY);
        if (meminfo_fd_ < 0) return;
    }
    if (lseek(meminfo_fd_, 0, SEEK_SET) < 0) return;

    char buf[4096];
    ssize_t n = read(meminfo_fd_, buf, sizeof(buf) - 1);
    if (n <= 0) return;
    buf[n] = '\0';

    int64_t total = 0, avail = 0, cached = 0, buffers = 0;

    char* p = buf;
    while (*p) {
        char* eol = strchr(p, '\n');
        if (!eol) break;

        int64_t val;
        if (sscanf(p, "MemTotal: %" SCNd64 " kB", &val) == 1) total = val;
        else if (sscanf(p, "MemAvailable: %" SCNd64 " kB", &val) == 1) avail = val;
        else if (sscanf(p, "Cached: %" SCNd64 " kB", &val) == 1) cached = val;
        else if (sscanf(p, "Buffers: %" SCNd64 " kB", &val) == 1) buffers = val;

        p = eol + 1;
    }

    stats_.total_kb = total;
    stats_.avail_kb = avail;
    stats_.used_kb = total - avail;
    stats_.cached_kb = cached + buffers;
    stats_.percent = total > 0 ? static_cast<int>(stats_.used_kb * 100 / total) : 0;
}
