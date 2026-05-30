#include "gpu.h"
#include "cpu.h"
#include "memory.h"
#include "network.h"
#include "power.h"
#include "renderer.h"
#include "themes.h"
#include "socket_api.h"
#include <csignal>
#include <unistd.h>
#include <poll.h>
#include <vector>

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) {
    g_running = 0;
}

// Wait for input on stdin (if tty) and/or socket, with timeout in ms.
// Returns true if stdin has data ready.
static const bool stdin_is_tty = isatty(STDIN_FILENO);

static bool wait_for_input(int listen_fd, int timeout_ms) {
    struct pollfd fds[2];
    int nfds = 0;
    int stdin_idx = -1;
    if (stdin_is_tty) {
        stdin_idx = nfds;
        fds[nfds] = {STDIN_FILENO, POLLIN, 0};
        nfds++;
    }
    if (listen_fd >= 0) {
        fds[nfds] = {listen_fd, POLLIN, 0};
        nfds++;
    }
    if (nfds == 0) { usleep(timeout_ms * 1000); return false; }
    int ret = poll(fds, nfds, timeout_ms);
    if (ret <= 0) return false;
    return stdin_idx >= 0 && (fds[stdin_idx].revents & POLLIN);
}

// Read an extended key: handles arrow key escape sequences.
static int read_key(Renderer& r) {
    int ch = r.poll_key();
    if (ch != 27) return ch;
    int ch2 = r.poll_key();
    if (ch2 != '[') return 27;
    int ch3 = r.poll_key();
    if (ch3 == 'A') return 256; // up
    if (ch3 == 'B') return 257; // down
    return 27;
}

static void gpu_selector_loop(Renderer& renderer, GpuMonitor& gpu, SocketApi& api,
                               std::vector<bool>& gpu_visible, const Theme& theme) {
    int cursor = 0;
    int count = gpu.count();
    if (count <= 0) return;
    gpu_visible.resize(count, true);
    renderer.clear_screen();

    while (g_running) {
        renderer.begin_frame();
        renderer.draw_gpu_selector(gpu, gpu_visible, cursor, theme);
        renderer.end_frame();

        while (g_running) {
            bool has_input = wait_for_input(api.fd(), 50);
            api.poll_clients();
            if (!has_input) continue;

            int ch = read_key(renderer);
            if (ch == -1) continue;
            bool redraw = true;
            switch (ch) {
                case 256: case 'k': case 'K':
                    cursor = (cursor - 1 + count) % count; break;
                case 257: case 'j': case 'J':
                    cursor = (cursor + 1) % count; break;
                case ' ': case '\r': case '\n':
                    gpu_visible[cursor] = !gpu_visible[cursor]; break;
                case 'a': case 'A':
                    for (size_t gi = 0; gi < gpu_visible.size(); gi++) gpu_visible[gi] = true;
                    break;
                case 'n':
                    for (size_t gi = 0; gi < gpu_visible.size(); gi++) gpu_visible[gi] = false;
                    break;
                case 'v': case 'V': case 27:
                    renderer.clear_screen(); return;
                default: redraw = false; break;
            }
            if (redraw) break;
        }
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    GpuMonitor gpu;
    CpuMonitor cpu;
    MemoryMonitor mem;
    NetworkMonitor net;
    PowerMonitor power;
    Renderer renderer;
    ThemeManager themes;
    SocketApi api;

    gpu.detect();
    cpu.init();
    net.detect();
    power.init();
    renderer.init();
    api.init();

    bool show_gpu = true, show_cpu = true, show_ram = true, show_net = true;
    unsigned frame = 0;
    std::vector<bool> gpu_visible(gpu.count(), true);

    static const int speed_us[] = {250000, 500000, 1000000, 2000000, 5000000, 10000000};
    static const char* speed_labels[] = {"0.25 second", "0.5 second", "1 second", "2 second", "5 second", "10 second"};
    static const int num_speeds = 6;
    int speed_idx = 2;

    while (g_running) {
        if (frame % 30 == 0) {
            gpu.detect();
            gpu_visible.resize(gpu.count(), true);
        }
        frame++;

        renderer.begin_frame();
        if (renderer.width() < 100) {
            renderer.draw_too_narrow(renderer.width());
            renderer.end_frame();
            usleep(1000000);
            continue;
        }

        power.poll();

        if (show_gpu) {
            int smi_interval = speed_us[speed_idx] >= 1000000 ? speed_us[speed_idx] / 1000000 : 1;
            gpu.poll_power_levels(smi_interval);
            gpu.poll();
            renderer.draw_gpu(gpu, themes.current(), gpu_visible);
        }
        if (show_cpu) {
            cpu.poll();
            renderer.draw_cpu(cpu, power.stats(), themes.current());
        }
        if (show_ram) {
            mem.poll();
            renderer.draw_ram(mem, themes.current());
        }
        if (show_net) {
            net.poll();
            renderer.draw_net(net, themes.current());
        }

        renderer.draw_status(show_gpu, show_cpu, show_ram, show_net,
                            themes, speed_labels[speed_idx]);
        renderer.end_frame();

        api.update_json(gpu, cpu, mem, net, power);

        // Single poll(): sleep for the full update interval, wake on keyboard or socket
        int timeout_ms = speed_us[speed_idx] / 1000;
        bool has_stdin = wait_for_input(api.fd(), timeout_ms);
        api.poll_clients();
        if (has_stdin) {
            int ch;
            while ((ch = renderer.poll_key()) != -1) {
                switch (ch) {
                    case 'g': case 'G': show_gpu = !show_gpu; renderer.clear_screen(); break;
                    case 'c': case 'C': show_cpu = !show_cpu; renderer.clear_screen(); break;
                    case 'r': case 'R': show_ram = !show_ram; renderer.clear_screen(); break;
                    case 'n': case 'N': show_net = !show_net; renderer.clear_screen(); break;
                    case 't': case 'T': themes.next(); renderer.clear_screen(); break;
                    case 's': case 'S': speed_idx = (speed_idx + 1) % num_speeds; break;
                    case 'v': case 'V':
                        gpu_selector_loop(renderer, gpu, api, gpu_visible, themes.current());
                        break;
                    case 'q': case 'Q': g_running = 0; break;
                }
            }
        }
    }

    api.shutdown();
    power.shutdown();
    renderer.shutdown();
    return 0;
}
