// Which T-Watch 2020 this is (ANYA_WATCH_V from platformio.ini).  LilyGo's
// library wants these in the sketch, before LilyGoWatch.h, not as global
// build flags: its own .cpp files must not see them.
#pragma once
#if ANYA_WATCH_V == 1
#define LILYGO_WATCH_2020_V1
#elif ANYA_WATCH_V == 2
#define LILYGO_WATCH_2020_V2
#else
#define LILYGO_WATCH_2020_V3
#endif
#define LILYGO_WATCH_LVGL
#include <LilyGoWatch.h>
