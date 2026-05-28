#include "network.h"
#include "sysfs.h"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

void NetworkMonitor::detect() {
    ifaces_.clear();
    cache_.invalidate();
    for (auto& entry : fs::directory_iterator("/sys/class/net")) {
        auto name = entry.path().filename().string();
        if (!sysfs::exists(entry.path().string() + "/device")) continue;

        auto base = entry.path().string() + "/statistics/";
        IfacePrev p;
        p.name = name;
        p.rx_bytes = static_cast<uint64_t>(sysfs::read_int(base + "rx_bytes", 0));
        p.tx_bytes = static_cast<uint64_t>(sysfs::read_int(base + "tx_bytes", 0));
        p.rx_packets = static_cast<uint64_t>(sysfs::read_int(base + "rx_packets", 0));
        p.tx_packets = static_cast<uint64_t>(sysfs::read_int(base + "tx_packets", 0));
        p.rx_max = 0; p.rx_min = -1;
        p.tx_max = 0; p.tx_min = -1;
        ifaces_.push_back(p);
    }
    std::sort(ifaces_.begin(), ifaces_.end(),
        [](const IfacePrev& a, const IfacePrev& b) { return a.name < b.name; });
}

void NetworkMonitor::poll() {
    stats_.resize(ifaces_.size());
    for (size_t i = 0; i < ifaces_.size(); i++) {
        auto& p = ifaces_[i];
        auto& s = stats_[i];
        s.name = p.name;

        auto base = "/sys/class/net/" + p.name;
        auto speed = cache_.read_int(base + "/speed", 0);
        s.link_speed_mbps = speed > 0 ? static_cast<int>(speed) : 0;

        auto stat_base = base + "/statistics/";
        auto cur_rx = static_cast<uint64_t>(cache_.read_int(stat_base + "rx_bytes", 0));
        auto cur_tx = static_cast<uint64_t>(cache_.read_int(stat_base + "tx_bytes", 0));
        auto cur_rx_p = static_cast<uint64_t>(cache_.read_int(stat_base + "rx_packets", 0));
        auto cur_tx_p = static_cast<uint64_t>(cache_.read_int(stat_base + "tx_packets", 0));

        // Wrap-safe deltas (compute bytes first, then multiply to avoid overflow)
        uint64_t rx_delta = cur_rx >= p.rx_bytes ? cur_rx - p.rx_bytes : 0;
        uint64_t tx_delta = cur_tx >= p.tx_bytes ? cur_tx - p.tx_bytes : 0;
        s.rx_bps = static_cast<int64_t>(rx_delta <= INT64_MAX / 8 ? rx_delta * 8 : INT64_MAX);
        s.tx_bps = static_cast<int64_t>(tx_delta <= INT64_MAX / 8 ? tx_delta * 8 : INT64_MAX);
        s.rx_pps = cur_rx_p >= p.rx_packets ? static_cast<int64_t>(cur_rx_p - p.rx_packets) : 0;
        s.tx_pps = cur_tx_p >= p.tx_packets ? static_cast<int64_t>(cur_tx_p - p.tx_packets) : 0;

        p.rx_bytes = cur_rx;
        p.tx_bytes = cur_tx;
        p.rx_packets = cur_rx_p;
        p.tx_packets = cur_tx_p;

        // Utilization %
        if (s.link_speed_mbps > 0) {
            auto link_bps = static_cast<int64_t>(s.link_speed_mbps) * 1000000;
            s.rx_pct = std::min(100, static_cast<int>(s.rx_bps * 100 / link_bps));
            s.tx_pct = std::min(100, static_cast<int>(s.tx_bps * 100 / link_bps));
        } else {
            s.rx_pct = 0;
            s.tx_pct = 0;
        }

        // Min/max tracking
        int rx_mbps = static_cast<int>(s.rx_bps / 1000000);
        int tx_mbps = static_cast<int>(s.tx_bps / 1000000);
        if (rx_mbps > p.rx_max) p.rx_max = rx_mbps;
        if (tx_mbps > p.tx_max) p.tx_max = tx_mbps;
        if (p.rx_min < 0 || rx_mbps < p.rx_min) p.rx_min = rx_mbps;
        if (p.tx_min < 0 || tx_mbps < p.tx_min) p.tx_min = tx_mbps;

        s.rx_max_mbps = p.rx_max;
        s.rx_min_mbps = p.rx_min;
        s.tx_max_mbps = p.tx_max;
        s.tx_min_mbps = p.tx_min;
    }
}
