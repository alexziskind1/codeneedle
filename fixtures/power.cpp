#include "power.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

void PowerMonitor::init() {
    if (system("command -v turbostat >/dev/null 2>&1 && sudo -n true 2>/dev/null") != 0)
        return;

    pipe_ = popen("sudo -n turbostat --quiet --Summary --show PkgWatt --interval 1 2>/dev/null", "r");
    if (!pipe_) return;

    int fd = fileno(pipe_);
    if (fd >= 0) fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

void PowerMonitor::poll() {
    if (!pipe_) return;

    char buf[256];
    while (fgets(buf, sizeof(buf), pipe_)) {
        if (strstr(buf, "PkgWatt")) continue;

        double pkg = 0;
        if (sscanf(buf, "%lf", &pkg) == 1) {
            stats_.pkg_watt = pkg;
            stats_.available = true;
        }
    }
    if (!feof(pipe_)) clearerr(pipe_);
}

void PowerMonitor::shutdown() {
    if (pipe_) {
        pclose(pipe_);
        pipe_ = nullptr;
    }
}
