#pragma once
#include "graphics_internal.hpp"

namespace application {

// ========================================================================
// Жизненный цикл лабораторной: создание, очистка, обновление, рисование
// ========================================================================
bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);
}
