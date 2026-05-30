#include "stats_service.h"
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

// gRPC-style service exposing the monitor snapshot over RPC. Exercises
// namespace-qualified return types and multi-line parameter lists.

::grpc::Status StatsService::GetSnapshot(
    ::grpc::ServerContext* context, const ::stats::SnapshotRequest* request,
    ::stats::SnapshotResponse* response) {
    std::lock_guard<std::mutex> guard(lock_);

    const auto now = std::chrono::steady_clock::now();
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_poll_);
    if (age.count() > stale_after_ms_) {
        return ::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE, "snapshot is stale");
    }

    response->set_timestamp_ms(last_timestamp_ms_);
    response->set_cpu_temp_c(snapshot_.cpu_temp_c);
    response->set_cpu_load_pct(snapshot_.cpu_load_pct);
    response->set_gpu_temp_c(snapshot_.gpu_temp_c);
    response->set_gpu_load_pct(snapshot_.gpu_load_pct);
    response->set_mem_used_kb(snapshot_.mem_used_kb);
    response->set_mem_total_kb(snapshot_.mem_total_kb);
    response->set_net_rx_bytes(snapshot_.net_rx_bytes);
    response->set_net_tx_bytes(snapshot_.net_tx_bytes);

    if (request->include_power()) {
        response->set_power_watts(snapshot_.power_watts);
    }

    served_++;
    return ::grpc::Status::OK;
}

::grpc::Status StatsService::ResetCounters(
    ::grpc::ServerContext* context,
    const ::google::protobuf::Empty* request,
    ::stats::ResetResponse* response) {
    std::lock_guard<std::mutex> guard(lock_);

    const uint64_t prior = served_;
    served_ = 0;
    errors_ = 0;
    last_reset_ms_ = last_timestamp_ms_;

    response->set_prior_served(prior);
    response->set_reset_at_ms(last_reset_ms_);
    return ::grpc::Status::OK;
}
