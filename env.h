#pragma once
#include "maze.h"
#include <tuple>
#include <cstdlib>
#include <cmath>

// Environment - optimized for tight loop performance
struct Env {
  Maze mz;
  int r{1}, c{1};
  int steps{0};
  int episode{0};

  // Constants for performance
  static constexpr int max_steps = 400;
  static constexpr float step_cost = -0.004f;
  static constexpr float wall_pen = -0.25f;
  static constexpr float goal_rew = 25.0f;

  inline void reset(bool regen = false) noexcept {
    if (regen)
      mz.generate(mz.H, mz.W);
    r = mz.start.first;
    c = mz.start.second;
    steps = 0;
    episode++;
  }

  // Step function - hot path, must be fast
  [[nodiscard]] inline std::tuple<int, int, float, bool>
  step(int a) noexcept {
    steps++;
    int nr = r, nc = c;

    // Branchless action decoding for performance
    nr -= (a == 0);
    nr += (a == 1);
    nc -= (a == 2);
    nc += (a == 3);

    float rew = step_cost;
    bool done = false;
    float prev_dist =
        std::abs(r - mz.goal.first) + std::abs(c - mz.goal.second);

    // Check wall collision
    if (mz.at(nr, nc) == 1) {
      rew += wall_pen;
      nr = r;
      nc = c;
    }

    r = nr;
    c = nc;

    float new_dist =
        std::abs(r - mz.goal.first) + std::abs(c - mz.goal.second);
    rew += 0.01f * (prev_dist - new_dist);

    // Check goal
    if (r == mz.goal.first && c == mz.goal.second) {
      rew += goal_rew;
      done = true;
    }

    if (steps >= max_steps)
      done = true;

    return {r, c, rew, done};
  }
};
