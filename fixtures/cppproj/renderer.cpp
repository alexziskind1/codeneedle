#include "renderer.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <clocale>
#include <string>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

// Frame-buffered ANSI rendering. All draw calls emit into frame_buf_.
// end_frame() does a single write() to stdout.

void Renderer::emit(const char* fmt, ...) {
    if (frame_off_ >= FRAME_BUF_SIZE - 1) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(frame_buf_ + frame_off_, FRAME_BUF_SIZE - frame_off_, fmt, ap);
    va_end(ap);
    if (n > 0) {
        frame_off_ += n;
        if (frame_off_ >= FRAME_BUF_SIZE) frame_off_ = FRAME_BUF_SIZE - 1;
    }
}

void Renderer::flush_frame() {
    if (frame_off_ > 0) {
        ssize_t ret __attribute__((unused)) = write(STDOUT_FILENO, frame_buf_, frame_off_);
        frame_off_ = 0;
    }
}

const char* Renderer::color_for_busy(int pct) {
    if (pct > 90) return "\033[0;31m";
    if (pct > 60) return "\033[1;33m";
    return "\033[0;32m";
}

const char* Renderer::color_for_temp(int temp) {
    if (temp > 85) return "\033[0;31m";
    if (temp > 70) return "\033[1;33m";
    return "\033[0;32m";
}

void Renderer::init() {
    setlocale(LC_ALL, "");

    tcgetattr(STDIN_FILENO, &orig_termios_);
    struct termios raw = orig_termios_;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    width_ = ws.ws_col > 0 ? ws.ws_col : 120;
    prev_width_ = 0;
    frame_off_ = 0;

    emit("\033[?25l\033[2J");
    flush_frame();
    rebuild_borders();
}

void Renderer::shutdown() {
    emit("\033[?25h\033[2J\033[H");
    flush_frame();
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios_);
}

int Renderer::poll_key() {
    char ch;
    if (read(STDIN_FILENO, &ch, 1) == 1)
        return ch;
    return -1;
}

void Renderer::clear_screen() {
    emit("\033[2J\033[H");
    flush_frame();
}

void Renderer::rebuild_borders() {
    int w = box_w() - 2;
    border_str_.clear();
    divider_str_.clear();
    for (int i = 0; i < w; i++) border_str_ += "═";
    for (int i = 0; i < w; i++) divider_str_ += "─";
}

void Renderer::begin_frame() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        width_ = ws.ws_col;
    resized_ = (width_ != prev_width_);
    frame_off_ = 0;
    if (resized_) {
        emit("\033[2J");
        prev_width_ = width_;
        rebuild_borders();
    }
    emit("\033[H");
}

void Renderer::end_frame() {
    emit("\033[J");
    flush_frame();
}

void Renderer::draw_too_narrow(int cols) {
    emit("\033[1;31mTerminal too narrow: %d cols. Minimum 100 required.\033[0m\n", cols);
    emit("\033[2mResize your terminal to continue.\033[0m");
}

std::string Renderer::make_bar_str(int pct, int width) {
    int filled = pct * width / 100;
    int empty = width - filled;
    std::string bar;
    bar += color_for_busy(pct);
    for (int i = 0; i < filled; i++) bar += "█";
    bar += "\033[2m";
    for (int i = 0; i < empty; i++) bar += "░";
    bar += "\033[0m";
    return bar;
}

int Renderer::net_bar_w() const {
    int w = inner() - 53;
    return w < 10 ? 10 : w;
}

void Renderer::draw_border_top(int color) {
    emit("\033[1m\033[38;5;%dm╔%s╗\033[0m\n", color, border_str_.c_str());
}

void Renderer::draw_border_mid(int color) {
    emit("\033[1m\033[38;5;%dm╠%s╣\033[0m\n", color, border_str_.c_str());
}

void Renderer::draw_border_bot(int color) {
    emit("\033[1m\033[38;5;%dm╚%s╝\033[0m\n", color, border_str_.c_str());
}

void Renderer::draw_divider(int color) {
    emit("\033[38;5;%dm╟%s╢\033[0m\n", color, divider_str_.c_str());
}

void Renderer::draw_title(const char* title, int color) {
    int pad = (inner() + static_cast<int>(strlen(title))) / 2;
    emit("\033[1m\033[38;5;%dm║\033[0m\033[1m\033[38;5;%dm%*s\033[%dG║\033[0m\n",
         color, color, pad, title, box_w());
}

// ── GPU ──

void Renderer::draw_gpu(const GpuMonitor& gpu, const Theme& theme, const std::vector<bool>& visible) {
    int c = theme.gpu_color;
    int rb = box_w();
    int gbw = gpu_bar_w();
    draw_border_top(c);
    draw_title("GPU System Monitor", c);
    draw_border_mid(c);

    auto& stats = gpu.stats();
    bool first = true;
    for (size_t i = 0; i < stats.size(); i++) {
        if (i < visible.size() && !visible[i]) continue;
        auto& s = stats[i];

        if (!first) draw_divider(c);
        first = false;

        auto bar = make_bar_str(s.usage, gbw);
        const char* gc = color_for_busy(s.usage);
        const char* tc = color_for_temp(s.temp_junction);

        emit("\033[38;5;%dm║\033[0m GPU%-2d %s%3d%%\033[0m %4d MHz %s  %s%3d/%3d/%3d\033[0m°C  %3d/%3dW  %4d/%3d%%"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, s.index, gc, s.usage, s.sclk_mhz, bar.c_str(),
             tc, s.temp_edge, s.temp_junction, s.temp_mem,
             s.power_avg_w, s.power_cap_w, s.fan_rpm, s.fan_percent, rb, c);

        auto vbar = make_bar_str(s.mem_percent, gbw);
        const char* vc = color_for_busy(s.mem_percent);

        emit("\033[38;5;%dm║\033[0m  VRAM %s%3d%%\033[0m %4d MHz %s",
             c, vc, s.mem_percent, s.mclk_mhz, vbar.c_str());

        if (!s.name.empty() || !s.power_level.empty()) {
            emit("  \033[2m%s", s.name.c_str());
            if (!s.power_level.empty()) {
                int plw = static_cast<int>(s.power_level.size()) + 2;
                emit("\033[%dG[%s]", rb - plw, s.power_level.c_str());
            }
            emit("\033[0m");
        }

        emit("\033[%dG\033[38;5;%dm║\033[0m\n", rb, c);
    }

    draw_border_bot(c);
}

void Renderer::draw_gpu_selector(const GpuMonitor& gpu, const std::vector<bool>& visible,
                                  int cursor, const Theme& theme) {
    int c = theme.gpu_color;
    int rb = box_w();
    draw_border_top(c);
    draw_title("GPU Display Selection", c);
    draw_border_mid(c);

    auto& stats = gpu.stats();
    for (size_t i = 0; i < stats.size(); i++) {
        auto& s = stats[i];
        bool vis = i < visible.size() ? visible[i] : true;
        const char* check = vis ? "X" : " ";
        const char* arrow = (static_cast<int>(i) == cursor) ? "\033[1m>\033[0m" : " ";

        emit("\033[38;5;%dm║\033[0m %s \033[1m[%s]\033[0m GPU%-2d  %5d MiB  %3d/%3d/%3d°C  %3d/%3dW"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, arrow, check, s.index, s.mem_total_mb,
             s.temp_edge, s.temp_junction, s.temp_mem,
             s.power_avg_w, s.power_cap_w, rb, c);
    }

    draw_border_mid(c);
    emit("\033[38;5;%dm║\033[0m \033[2m\xe2\x86\x91\xe2\x86\x93 Navigate  Space Toggle  A All  N None  V Exit\033[0m"
         "\033[%dG\033[38;5;%dm║\033[0m\n", c, rb, c);
    draw_border_bot(c);
}

// ── CPU ──

void Renderer::draw_cpu(const CpuMonitor& cpu, const PowerStats& power, const Theme& theme) {
    int c = theme.cpu_color;
    int rb = box_w();
    draw_border_top(c);
    draw_title("CPU System Monitor", c);
    draw_border_mid(c);

    auto& sum = cpu.summary();
    const char* lc = color_for_busy(sum.avg_busy);
    const char* ptc = color_for_temp(sum.pkg_temp);
    const char* c1c = color_for_temp(sum.ccd1_temp);
    const char* c2c = color_for_temp(sum.ccd2_temp);

    if (power.available) {
        emit("\033[38;5;%dm║\033[0m \033[1mLoad:\033[0m %s%3d%%\033[0m  \033[1mAvg:\033[0m %4d MHz  "
             "\033[1mPkg:\033[0m %s%d°C\033[0m  \033[1mCCD1:\033[0m %s%d°C\033[0m  "
             "\033[1mCCD2:\033[0m %s%d°C\033[0m  \033[1mPower:\033[0m %.1fW"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, lc, sum.avg_busy, sum.avg_mhz,
             ptc, sum.pkg_temp, c1c, sum.ccd1_temp, c2c, sum.ccd2_temp,
             power.pkg_watt, rb, c);
    } else {
        emit("\033[38;5;%dm║\033[0m \033[1mLoad:\033[0m %s%3d%%\033[0m  \033[1mAvg:\033[0m %4d MHz  "
             "\033[1mPkg:\033[0m %s%d°C\033[0m  \033[1mCCD1:\033[0m %s%d°C\033[0m  "
             "\033[1mCCD2:\033[0m %s%d°C\033[0m  \033[1mPower:\033[0m N/A"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, lc, sum.avg_busy, sum.avg_mhz,
             ptc, sum.pkg_temp, c1c, sum.ccd1_temp, c2c, sum.ccd2_temp, rb, c);
    }

    draw_border_mid(c);

    auto& cores = cpu.cores();
    int nc = static_cast<int>(cores.size());
    int half = (nc + 1) / 2;
    int bw = core_bar_w();

    for (int row = 0; row < half; row++) {
        auto& left = cores[row];
        auto lbar = make_bar_str(left.busy_percent, bw);
        auto lbc = color_for_busy(left.busy_percent);

        int ri = row + half;
        if (ri < nc) {
            auto& right = cores[ri];
            auto rbar = make_bar_str(right.busy_percent, bw);
            auto rbc = color_for_busy(right.busy_percent);
            emit("\033[38;5;%dm║\033[0m  C%-2d %s%3d%%\033[0m %4d MHz %s  │  C%-2d %s%3d%%\033[0m %4d MHz %s"
                 "\033[%dG\033[38;5;%dm║\033[0m\n",
                 c, left.core_id, lbc, left.busy_percent, left.freq_mhz, lbar.c_str(),
                 right.core_id, rbc, right.busy_percent, right.freq_mhz, rbar.c_str(), rb, c);
        } else {
            emit("\033[38;5;%dm║\033[0m  C%-2d %s%3d%%\033[0m %4d MHz %s  │"
                 "\033[%dG\033[38;5;%dm║\033[0m\n",
                 c, left.core_id, lbc, left.busy_percent, left.freq_mhz, lbar.c_str(), rb, c);
        }
    }

    draw_border_bot(c);
}

// ── RAM ──

void Renderer::draw_ram(const MemoryMonitor& mem, const Theme& theme) {
    int c = theme.ram_color;
    int rb = box_w();
    draw_border_top(c);
    draw_title("RAM System Monitor", c);
    draw_border_mid(c);

    auto& s = mem.stats();
    int t_w = static_cast<int>(s.total_kb * 10 / 1048576);
    int u_w = static_cast<int>(s.used_kb * 10 / 1048576);
    int a_w = static_cast<int>(s.avail_kb * 10 / 1048576);
    int c_w = static_cast<int>(s.cached_kb * 10 / 1048576);
    auto mc = color_for_busy(s.percent);

    emit("\033[38;5;%dm║\033[0m \033[1mPhysical:\033[0m %3d.%d GiB   "
         "\033[1mUsed:\033[0m %s%3d.%d GiB\033[0m   "
         "\033[1mAvailable:\033[0m %3d.%d GiB   "
         "\033[1mCached:\033[0m %3d.%d GiB"
         "\033[%dG\033[38;5;%dm║\033[0m\n",
         c, t_w/10, t_w%10, mc, u_w/10, u_w%10, a_w/10, a_w%10, c_w/10, c_w%10, rb, c);

    draw_border_mid(c);

    auto bar = make_bar_str(s.percent, ram_bar_w());
    emit("\033[38;5;%dm║\033[0m %s%3d%%\033[0m %s\033[%dG\033[38;5;%dm║\033[0m\n",
         c, mc, s.percent, bar.c_str(), rb, c);

    draw_border_bot(c);
}

// ── NET ──

std::string format_bps(int64_t bps) {
    char buf[32];
    if (bps >= 1000000000)
        snprintf(buf, sizeof(buf), "%ld.%ld Gbps", bps/1000000000, (bps/100000000)%10);
    else if (bps >= 1000000)
        snprintf(buf, sizeof(buf), "%ld.%ld Mbps", bps/1000000, (bps/100000)%10);
    else if (bps >= 1000)
        snprintf(buf, sizeof(buf), "%ld.%ld Kbps", bps/1000, (bps/100)%10);
    else
        snprintf(buf, sizeof(buf), "%ld bps", bps);
    return buf;
}

void Renderer::draw_net(const NetworkMonitor& net, const Theme& theme) {
    int c = theme.net_color;
    int rb = box_w();
    int nbw = net_bar_w();
    draw_border_top(c);
    draw_title("NET System Monitor", c);
    draw_border_mid(c);

    auto& stats = net.stats();
    for (size_t i = 0; i < stats.size(); i++) {
        auto& s = stats[i];
        auto rx_str = format_bps(s.rx_bps);
        auto tx_str = format_bps(s.tx_bps);
        auto rx_bar = make_bar_str(s.rx_pct, nbw);
        auto tx_bar = make_bar_str(s.tx_pct, nbw);

        emit("\033[38;5;%dm║\033[0m \033[1m%-10s\033[0m Link: %d Mbps   Pkts: RX %ld/s  TX %ld/s"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, s.name.c_str(), s.link_speed_mbps, s.rx_pps, s.tx_pps, rb, c);
        emit("\033[38;5;%dm║\033[0m   \033[1mRX\033[0m %3d%% %s %-14s Min: %4d  Max: %4d Mbps"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, s.rx_pct, rx_bar.c_str(), rx_str.c_str(), s.rx_min_mbps, s.rx_max_mbps, rb, c);
        emit("\033[38;5;%dm║\033[0m   \033[1mTX\033[0m %3d%% %s %-14s Min: %4d  Max: %4d Mbps"
             "\033[%dG\033[38;5;%dm║\033[0m\n",
             c, s.tx_pct, tx_bar.c_str(), tx_str.c_str(), s.tx_min_mbps, s.tx_max_mbps, rb, c);

        if (i + 1 < stats.size()) draw_divider(c);
    }

    draw_border_bot(c);
}

// ── Status bar ──

void Renderer::draw_status(bool show_gpu, bool show_cpu, bool show_ram, bool show_net,
                            const ThemeManager& themes, const char* speed_label) {
    const char* on = "\033[0;32m";
    const char* off = "\033[2m";
    auto& t = themes.current();
    emit(" %s[G]GPU\033[0m  %s[C]CPU\033[0m  %s[R]RAM\033[0m  %s[N]NET\033[0m  "
         "\033[1;33m[T]%s\033[0m  \033[0;36m[S]%s\033[0m  \033[2m[Q]Quit\033[0m\n",
         show_gpu ? on : off, show_cpu ? on : off,
         show_ram ? on : off, show_net ? on : off, t.name.c_str(), speed_label);
}
