#include "stats_service.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Collection of real-world multi-line signature shapes the extractor must
// handle: trailing return types, return-type-on-its-own-line, default
// arguments, noexcept/const qualifiers, templated returns, gRPC streaming,
// and constructor member-initializer lists.

// 1. Constructor with a multi-line parameter list and member init list.
StatsService::StatsService(
    const ServiceConfig& cfg,
    std::shared_ptr<Clock> clock)
    : cfg_(cfg),
      clock_(std::move(clock)),
      stale_after_ms_(cfg.stale_after_ms),
      served_(0),
      errors_(0),
      last_timestamp_ms_(0),
      last_reset_ms_(0) {
    snapshot_ = Snapshot{};
    snapshot_.cpu_temp_c = 0.0;
    snapshot_.cpu_load_pct = 0.0;
    snapshot_.gpu_temp_c = 0.0;
    snapshot_.gpu_load_pct = 0.0;
    snapshot_.mem_total_kb = cfg.mem_total_kb;
    snapshot_.mem_used_kb = 0;
    snapshot_.net_rx_bytes = 0;
    snapshot_.net_tx_bytes = 0;
    snapshot_.power_watts = 0.0;
    last_poll_ = clock_->now();
    if (cfg.warm_start) {
        served_ = cfg.prior_served;
        errors_ = cfg.prior_errors;
        last_reset_ms_ = cfg.prior_reset_ms;
        last_timestamp_ms_ = cfg.prior_timestamp_ms;
    }
    history_.reserve(cfg.history_capacity);
    history_.clear();
    buckets_.clear();
    if (cfg.stale_after_ms <= 0) {
        stale_after_ms_ = kDefaultStaleMs;
    }
}

// 2. Trailing return type (auto ... -> T) with a multi-line parameter list.
auto StatsService::ComputeRollup(
    const std::vector<Sample>& samples,
    std::size_t window) -> RollupResult {
    RollupResult out{};
    if (samples.empty() || window == 0) {
        return out;
    }

    const std::size_t span = std::min(window, samples.size());
    double sum = 0.0;
    double peak = samples.back().value;
    double trough = samples.back().value;

    for (std::size_t i = samples.size() - span; i < samples.size(); ++i) {
        const double v = samples[i].value;
        sum += v;
        peak = std::max(peak, v);
        trough = std::min(trough, v);
    }

    out.count = span;
    out.mean = sum / static_cast<double>(span);
    out.peak = peak;
    out.trough = trough;
    out.range = peak - trough;
    return out;
}

// 3. Return type on its own line, multi-line params, trailing const qualifier,
//    and the opening brace on its own line.
std::map<std::string, std::uint64_t>
StatsService::HistogramBuckets(
    const std::vector<double>& values,
    double bucket_width) const
{
    std::map<std::string, std::uint64_t> hist;
    if (bucket_width <= 0.0) {
        return hist;
    }

    std::uint64_t overflow = 0;
    for (double v : values) {
        if (!std::isfinite(v)) {
            overflow++;
            continue;
        }
        const long idx = static_cast<long>(std::floor(v / bucket_width));
        const double lo = static_cast<double>(idx) * bucket_width;
        const double hi = lo + bucket_width;
        char label[64];
        std::snprintf(label, sizeof(label), "%.1f-%.1f", lo, hi);
        hist[std::string(label)] += 1;
    }

    if (overflow > 0) {
        hist["nan"] = overflow;
    }
    std::uint64_t peak = 0;
    for (const auto& kv : hist) {
        peak = std::max(peak, kv.second);
    }
    hist["__peak__"] = peak;
    if (hist.empty()) {
        hist["empty"] = 0;
    }
    return hist;
}

// 4. gRPC streaming RPC: default arguments spanning lines and a templated
//    writer parameter with nested namespace qualifiers.
::grpc::Status StatsService::StreamSamples(
    ::grpc::ServerContext* context,
    const ::stats::StreamRequest* request,
    ::grpc::ServerWriter<::stats::Sample>* writer) {
    if (writer == nullptr) {
        return ::grpc::Status(
            ::grpc::StatusCode::INVALID_ARGUMENT, "writer is null");
    }

    std::lock_guard<std::mutex> guard(lock_);
    const std::size_t limit = request->max_samples() > 0
        ? static_cast<std::size_t>(request->max_samples())
        : history_.size();

    std::size_t emitted = 0;
    for (const Sample& s : history_) {
        if (emitted >= limit) {
            break;
        }
        ::stats::Sample wire;
        wire.set_timestamp_ms(s.timestamp_ms);
        wire.set_value(s.value);
        if (!writer->Write(wire)) {
            errors_++;
            return ::grpc::Status(
                ::grpc::StatusCode::ABORTED, "client closed stream");
        }
        emitted++;
    }
    return ::grpc::Status::OK;
}

// 5. Templated return type, multi-line params, noexcept qualifier.
std::vector<std::pair<std::string, double>>
StatsService::TopContributors(
    const std::map<std::string, double>& weights,
    std::size_t top_n) const noexcept {
    std::vector<std::pair<std::string, double>> ranked;
    ranked.reserve(weights.size());
    for (const auto& kv : weights) {
        ranked.emplace_back(kv.first, kv.second);
    }

    std::sort(ranked.begin(), ranked.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    if (ranked.size() > top_n) {
        ranked.resize(top_n);
    }

    double total = 0.0;
    for (const auto& kv : ranked) {
        total += kv.second;
    }
    if (total > 0.0) {
        for (auto& kv : ranked) {
            kv.second = kv.second / total;
        }
    }
    return ranked;
}
