#include "themes.h"

ThemeManager::ThemeManager() {
    themes_ = {{
        {"Default",  36,  213, 208, 114},   // cyan/pink/orange/green
        {"Vivid",    39,  196, 226, 46},     // blue/red/yellow/green
        {"Mono",     245, 250, 240, 255},    // grays
        {"Ocean",    51,  49,  43,  87},     // teals/aquas
        {"Lava",     202, 196, 220, 214},    // reds/oranges
        {"Pastel",   141, 183, 219, 147},    // soft purples/pinks
        {"Matrix",   46,  34,  118, 82},     // all greens
        {"Indigo",   69,  105, 63,  99},     // purples/blues
        {"Warm",     216, 180, 223, 174},    // peach/tan
        {"Neon",     197, 171, 207, 163},    // hot pinks
    }};
}

void ThemeManager::next() {
    index_ = (index_ + 1) % static_cast<int>(themes_.size());
}
