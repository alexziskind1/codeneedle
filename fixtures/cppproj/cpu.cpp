#include "cpu.h"
#include "sysfs.h"
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <set>
#include <thread>
#include <unistd.h>
#include <fcntl.h>

namespace fs = std::filesystem;

CpuMonitor::~CpuMonitor() {
    if (stat_fd_ >= 0) close(stat_fd_);
}

void CpuMonitor::init() {
    detect_topology();
    detect_temp_sensors();
    stat_fd_ = open("/proc/stat", O_RDONLY);
    // Take initial /proc/stat snapshot
    poll();
}

void CpuMonitor::detect_topology() {
    num_cpus_ = static_cast<int>(std::thread::hardware_concurrency());
    if (num_cpus_ <= 0) num_cpus_ = 1;
    std::set<int> core_set;

    for (int i = 0; i < num_cpus_; i++) {
        auto path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id";
        int core_id = static_cast<int>(sysfs::read_int(path, i));
        cpu_to_core_[i] = core_id;
        if (core_set.insert(core_id).second) {
            core_first_cpu_[core_id] = i;
        }
    }

    sorted_cores_.assign(core_set.begin(), core_set.end());
}

void CpuMonitor::detect_temp_sensors() {
    try {
        for (auto& entry : fs::directory_iterator("/sys/class/hwmon")) {
            auto name = sysfs::read_file(entry.path().string() + "/name");
            if (name == "k10temp" || name == "coretemp") {
                hwmon_dir_ = entry.path().string();
                break;
            }
        }
    } catch (...) { return; }

    if (hwmon_dir_.empty()) return;

    try {
        for (auto& entry : fs::directory_iterator(hwmon_dir_)) {
            auto fname = entry.path().filename().string();
            if (fname.find("_label") == std::string::npos) continue;
            auto label = sysfs::read_file(entry.path().string());
            auto input = entry.path().string();
            auto pos = input.rfind("_label");
            input.replace(pos, 6, "_input");
            if (label == "Tccd1") ccd1_path_ = input;
            else if (label == "Tccd2") ccd2_path_ = input;
        }
    } catch (...) {}
}

void CpuMonitor::poll() {
    if (stat_fd_ < 0) return;
    if (lseek(stat_fd_, 0, SEEK_SET) < 0) return;

    char stat_buf[8192];
    ssize_t n = read(stat_fd_, stat_buf, sizeof(stat_buf) - 1);
    if (n <= 0) return;
    stat_buf[n] = '\0';

    // Per-core accumulators
    std::unordered_map<int, int64_t> core_busy_sum;
    std::unordered_map<int, int> core_busy_cnt;
    for (auto c : sorted_cores_) {
        core_busy_sum[c] = 0;
        core_busy_cnt[c] = 0;
    }

    char* p = stat_buf;
    while (*p) {
        char* eol = strchr(p, '\n');
        if (!eol) break;

        if (p[0] == 'c' && p[1] == 'p' && p[2] == 'u' && p[3] >= '0' && p[3] <= '9') {
            int cpuid;
            uint64_t user, nice, sys, idle, iowait, irq, softirq, steal;
            if (sscanf(p, "cpu%d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64
                       " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
                       &cpuid, &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal) >= 9) {

                uint64_t cur_idle = idle + iowait;
                uint64_t cur_total = user + nice + sys + idle + iowait + irq + softirq + steal;

                int busy = 0;
                auto it = prev_.find(cpuid);
                if (it != prev_.end() && cur_total >= it->second.total) {
                    uint64_t dt = cur_total - it->second.total;
                    uint64_t di = cur_idle >= it->second.idle ? cur_idle - it->second.idle : 0;
                    if (dt > 0 && di <= dt) busy = static_cast<int>((dt - di) * 100 / dt);
                }
                prev_[cpuid] = {cur_total, cur_idle};

                auto cit = cpu_to_core_.find(cpuid);
                if (cit != cpu_to_core_.end()) {
                    core_busy_sum[cit->second] += busy;
                    core_busy_cnt[cit->second]++;
                }
            }
        }

        p = eol + 1;
    }

    // Build per-core stats
    cores_.resize(sorted_cores_.size());
    int64_t total_busy = 0, total_mhz = 0;

    for (size_t i = 0; i < sorted_cores_.size(); i++) {
        int c = sorted_cores_[i];
        auto& cs = cores_[i];
        cs.core_id = c;

        int cnt = core_busy_cnt[c];
        cs.busy_percent = cnt > 0 ? static_cast<int>(core_busy_sum[c] / cnt) : 0;

        auto freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(core_first_cpu_[c])
                         + "/cpufreq/scaling_cur_freq";
        cs.freq_mhz = static_cast<int>(cache_.read_int(freq_path, 0) / 1000);

        total_busy += cs.busy_percent;
        total_mhz += cs.freq_mhz;
    }

    int nc = static_cast<int>(sorted_cores_.size());
    summary_.avg_busy = nc > 0 ? static_cast<int>(total_busy / nc) : 0;
    summary_.avg_mhz = nc > 0 ? static_cast<int>(total_mhz / nc) : 0;

    // Temps
    summary_.pkg_temp = 0;
    summary_.ccd1_temp = 0;
    summary_.ccd2_temp = 0;
    if (!hwmon_dir_.empty()) {
        summary_.pkg_temp = static_cast<int>(cache_.read_int(hwmon_dir_ + "/temp1_input", 0) / 1000);
        if (!ccd1_path_.empty()) summary_.ccd1_temp = static_cast<int>(cache_.read_int(ccd1_path_, 0) / 1000);
        if (!ccd2_path_.empty()) summary_.ccd2_temp = static_cast<int>(cache_.read_int(ccd2_path_, 0) / 1000);
    }
}
