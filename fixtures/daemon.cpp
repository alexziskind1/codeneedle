#include "gpu.h"
#include "cpu.h"
#include "memory.h"
#include "network.h"
#include "power.h"
#include "socket_api.h"
#include <csignal>
#include <cstdio>
#include <poll.h>

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) {
    g_running = 0;
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    GpuMonitor gpu;
    CpuMonitor cpu;
    MemoryMonitor mem;
    NetworkMonitor net;
    PowerMonitor power;
    SocketApi api;

    gpu.detect();
    cpu.init();
    net.detect();
    power.init();

    if (!api.init()) {
        fprintf(stderr, "systopd: failed to create socket at /tmp/systop.sock\n");
        return 1;
    }

    fprintf(stderr, "systopd: listening on /tmp/systop.sock\n");

    unsigned frame = 0;
    while (g_running) {
        if (frame % 30 == 0) gpu.detect();
        frame++;

        power.poll();
        gpu.poll_power_levels(1);
        gpu.poll();
        cpu.poll();
        mem.poll();
        net.poll();

        api.update_json(gpu, cpu, mem, net, power);

        // Single poll(): sleep 1s, wake on socket client connection
        struct pollfd pfd = {api.fd(), POLLIN, 0};
        int remaining_ms = 1000;
        while (remaining_ms > 0 && g_running) {
            int ret = poll(&pfd, 1, remaining_ms);
            if (ret > 0 && (pfd.revents & POLLIN)) {
                api.poll_clients();
                remaining_ms = 0; // served client, can proceed to next frame or wait more
            } else {
                break; // timeout — next frame
            }
        }
    }

    api.shutdown();
    power.shutdown();
    fprintf(stderr, "systopd: stopped\n");
    return 0;
}
