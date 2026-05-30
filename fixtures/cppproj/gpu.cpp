#include "gpu.h"
#include "sysfs.h"
#include <filesystem>
#include <algorithm>
#include <regex>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>

namespace fs = std::filesystem;

GpuMonitor::~GpuMonitor() {
    if (smi_pipe_) pclose(smi_pipe_);
}

void GpuMonitor::detect() {
    cards_.clear();
    cache_.invalidate();
    static const std::regex card_re("^card[0-9]+$");

    try {
        for (auto& entry : fs::directory_iterator("/sys/class/drm")) {
            auto name = entry.path().filename().string();
            if (!std::regex_match(name, card_re)) continue;
            auto drm = entry.path().string();
            if (!sysfs::exists(drm + "/device/gpu_busy_percent")) continue;

            std::string hwmon;
            auto hwmon_base = drm + "/device/hwmon";
            if (fs::exists(hwmon_base)) {
                for (auto& h : fs::directory_iterator(hwmon_base)) {
                    hwmon = h.path().string();
                    break;
                }
            }

            cards_.push_back({drm, hwmon, {}});
        }
    } catch (...) {}

    std::sort(cards_.begin(), cards_.end(),
        [](const CardInfo& a, const CardInfo& b) { return a.drm_path < b.drm_path; });

    // One-time rocm-smi for marketing names
    if (!names_loaded_) {
        FILE* p = popen("rocm-smi --showproductname 2>/dev/null", "r");
        if (p) {
            char line[256];
            while (fgets(line, sizeof(line), p)) {
                char* gpu_tag = strstr(line, "GPU[");
                char* series_tag = strstr(line, "Card Series:");
                if (gpu_tag && series_tag) {
                    int idx = atoi(gpu_tag + 4);
                    char* val = series_tag + 12;
                    while (*val == ' ' || *val == '\t') val++;
                    char* end = val + strlen(val) - 1;
                    while (end > val && (*end == ' ' || *end == '\n' || *end == '\r')) *end-- = '\0';
                    if (idx >= 0 && idx < static_cast<int>(cards_.size()))
                        name_cache_[cards_[idx].drm_path] = val;
                }
            }
            pclose(p);
        }
        names_loaded_ = true;
    }

    // Apply cached names
    for (auto& c : cards_) {
        auto it = name_cache_.find(c.drm_path);
        if (it != name_cache_.end())
            c.marketing_name = it->second;
    }

    power_levels_.resize(cards_.size());
}

void GpuMonitor::poll() {
    stats_.resize(cards_.size());
    for (size_t i = 0; i < cards_.size(); i++) {
        auto& c = cards_[i];
        auto& s = stats_[i];
        s.index = static_cast<int>(i);
        s.valid = !c.hwmon_path.empty();
        s.name = c.marketing_name;
        s.power_level = (i < power_levels_.size()) ? power_levels_[i] : "";

        auto dev = c.drm_path + "/device/";
        s.usage = static_cast<int>(cache_.read_int(dev + "gpu_busy_percent", 0));

        auto mem_used = cache_.read_int(dev + "mem_info_vram_used", 0);
        auto mem_total = cache_.read_int(dev + "mem_info_vram_total", 1);
        s.mem_used_mb = static_cast<int>(mem_used / 1048576);
        s.mem_total_mb = static_cast<int>(mem_total / 1048576);
        s.mem_percent = mem_total > 0 ? static_cast<int>(mem_used * 100 / mem_total) : 0;
        if (s.mem_percent > 100) s.mem_percent = 100;

        if (s.valid) {
            auto& h = c.hwmon_path;
            s.temp_edge    = static_cast<int>(cache_.read_int(h + "/temp1_input", 0) / 1000);
            s.temp_junction= static_cast<int>(cache_.read_int(h + "/temp2_input", 0) / 1000);
            s.temp_mem     = static_cast<int>(cache_.read_int(h + "/temp3_input", 0) / 1000);
            s.sclk_mhz     = static_cast<int>(cache_.read_int(h + "/freq1_input", 0) / 1000000);
            s.mclk_mhz     = static_cast<int>(cache_.read_int(h + "/freq2_input", 0) / 1000000);
            s.power_avg_w   = static_cast<int>(cache_.read_int(h + "/power1_average", 0) / 1000000);
            s.power_cap_w   = static_cast<int>(cache_.read_int(h + "/power1_cap", 0) / 1000000);
            s.fan_rpm       = static_cast<int>(cache_.read_int(h + "/fan1_input", 0));
            auto fan_max    = cache_.read_int(h + "/fan1_max", 6000);
            s.fan_percent   = fan_max > 0 ? static_cast<int>(s.fan_rpm * 100 / fan_max) : 0;
        }
    }
}

void GpuMonitor::poll_power_levels(int min_interval_s) {
    if (cards_.empty()) return;

    // If pipe is open, drain available output
    if (smi_pipe_) {
        char line[256];
        while (fgets(line, sizeof(line), smi_pipe_)) {
            int len = static_cast<int>(strlen(line));
            if (smi_off_ + len < static_cast<int>(sizeof(smi_buf_)) - 1) {
                memcpy(smi_buf_ + smi_off_, line, len);
                smi_off_ += len;
            }
        }
        if (feof(smi_pipe_)) {
            smi_buf_[smi_off_] = '\0';
            parse_smi_output();
            pclose(smi_pipe_);
            smi_pipe_ = nullptr;
            smi_off_ = 0;
            smi_last_ = time(nullptr);
        } else {
            clearerr(smi_pipe_);
        }
        return;
    }

    // Check if it's time to start a new poll
    time_t now = time(nullptr);
    if (now - smi_last_ < min_interval_s) return;

    smi_pipe_ = popen("rocm-smi --showperflevel 2>/dev/null", "r");
    if (!smi_pipe_) return;
    int fd = fileno(smi_pipe_);
    if (fd >= 0) fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    smi_off_ = 0;
}

void GpuMonitor::parse_smi_output() {
    power_levels_.resize(cards_.size());
    char* p = smi_buf_;
    while (*p) {
        char* eol = strchr(p, '\n');
        if (!eol) break;
        *eol = '\0';

        char* gpu_tag = strstr(p, "GPU[");
        char* level_tag = strstr(p, "Performance Level:");
        if (gpu_tag && level_tag) {
            int idx = atoi(gpu_tag + 4);
            char* val = level_tag + 18;
            while (*val == ' ') val++;
            // Trim trailing whitespace
            char* end = val + strlen(val) - 1;
            while (end > val && (*end == ' ' || *end == '\r')) *end-- = '\0';
            if (idx >= 0 && idx < static_cast<int>(power_levels_.size()))
                power_levels_[idx] = val;
        }

        p = eol + 1;
    }
}
