#include "sysfs.h"
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>

namespace sysfs {

std::string read_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return {};
    char buf[256];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return {};
    buf[n] = '\0';
    char* nl = strchr(buf, '\n');
    if (nl) *nl = '\0';
    return buf;
}

int64_t read_int(const std::string& path, int64_t fallback) {
    auto s = read_file(path);
    if (s.empty()) return fallback;
    char* end;
    long long val = strtoll(s.c_str(), &end, 10);
    if (end == s.c_str()) return fallback;
    return static_cast<int64_t>(val);
}

bool exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// --- CachedReader ---

CachedReader::~CachedReader() {
    invalidate();
}

void CachedReader::invalidate() {
    for (auto& kv : fds_) {
        if (kv.second >= 0) close(kv.second);
    }
    fds_.clear();
}

int CachedReader::get_fd(const std::string& path) {
    auto it = fds_.find(path);
    if (it != fds_.end()) {
        if (it->second >= 0) return it->second;
        return -1;
    }
    int fd = open(path.c_str(), O_RDONLY);
    fds_[path] = fd;
    return fd;
}

int64_t CachedReader::read_int(const std::string& path, int64_t fallback) {
    int fd = get_fd(path);
    if (fd < 0) return fallback;

    if (lseek(fd, 0, SEEK_SET) < 0) return fallback;

    char buf[64];
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) return fallback;
    buf[n] = '\0';

    char* end;
    long long val = strtoll(buf, &end, 10);
    if (end == buf) return fallback;
    return static_cast<int64_t>(val);
}

} // namespace sysfs
