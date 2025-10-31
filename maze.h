#pragma once
#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

// RNG utilities - inline for zero overhead
inline std::mt19937 &get_rng() {
  static std::mt19937 rng{std::random_device{}()};
  return rng;
}

inline int irand(int a, int b) {
  std::uniform_int_distribution<int> d(a, b);
  return d(get_rng());
}

inline float frand() {
  std::uniform_real_distribution<float> d(0.f, 1.f);
  return d(get_rng());
}

// Maze structure - optimized for cache efficiency
struct Maze {
  int H{17}, W{25};
  std::vector<uint8_t> grid;
  std::pair<int, int> start{1, 1}, goal{15, 23};

  // Inline accessors for zero overhead
  [[nodiscard]] inline uint8_t &at(int r, int c) noexcept {
    return grid[r * W + c];
  }

  [[nodiscard]] inline uint8_t at(int r, int c) const noexcept {
    return grid[r * W + c];
  }

  // Generate maze using DFS
  inline void generate(int h = 17, int w = 25) {
    H = (h % 2 ? h : h + 1);
    W = (w % 2 ? w : w + 1);
    grid.assign(H * W, 1);
    start = {1, 1};
    goal = {H - 2, W - 2};

    auto inside = [&](int r, int c) constexpr noexcept -> bool {
      return r > 0 && r < H - 1 && c > 0 && c < W - 1;
    };

    std::vector<std::pair<int, int>> stack;
    stack.reserve(H * W / 4);
    stack.emplace_back(1, 1);
    at(1, 1) = 0;

    while (!stack.empty()) {
      auto [r, c] = stack.back();

      // Check neighbors - unrolled for performance
      std::pair<int, int> dirs[4] = {{2, 0}, {-2, 0}, {0, 2}, {0, -2}};
      std::pair<int, int> valid[4];
      int valid_count = 0;

      for (auto [dr, dc] : dirs) {
        int nr = r + dr, nc = c + dc;
        if (inside(nr, nc) && at(nr, nc) == 1) {
          valid[valid_count++] = {dr, dc};
        }
      }

      if (valid_count == 0) {
        stack.pop_back();
        continue;
      }

      auto [dr, dc] = valid[irand(0, valid_count - 1)];
      int nr = r + dr, nc = c + dc;
      at(r + dr / 2, c + dc / 2) = 0;
      at(nr, nc) = 0;
      stack.emplace_back(nr, nc);
    }

    at(start.first, start.second) = 0;
    at(goal.first, goal.second) = 0;
  }
};
