#pragma once
#include <algorithm>
#include <cstdint>
#include <queue>
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

// Maze generation algorithms
enum class MazeAlgorithm {
  DFS_BACKTRACK,  // Recursive backtracking (long corridors)
  PRIMS,          // Randomized Prim's (more branching)
  HYBRID          // Mix of both
};

// Maze structure - optimized for cache efficiency
struct Maze {
  int H{21}, W{31};
  std::vector<uint8_t> grid;
  std::pair<int, int> start{1, 1}, goal{19, 29};
  MazeAlgorithm algorithm{MazeAlgorithm::HYBRID};

  // Inline accessors for zero overhead
  [[nodiscard]] inline uint8_t &at(int r, int c) noexcept {
    return grid[r * W + c];
  }

  [[nodiscard]] inline uint8_t at(int r, int c) const noexcept {
    return grid[r * W + c];
  }

  [[nodiscard]] inline bool inside(int r, int c) const noexcept {
    return r > 0 && r < H - 1 && c > 0 && c < W - 1;
  }

  // Generate maze using DFS (creates long corridors)
  inline void generate_dfs() noexcept {
    std::vector<std::pair<int, int>> stack;
    stack.reserve(H * W / 4);
    stack.emplace_back(1, 1);
    at(1, 1) = 0;

    while (!stack.empty()) {
      auto [r, c] = stack.back();

      // Check neighbors
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
  }

  // Generate maze using Prim's algorithm (more branching)
  inline void generate_prims() noexcept {
    std::vector<std::pair<int, int>> walls;
    walls.reserve(H * W / 2);

    // Start from (1, 1)
    at(1, 1) = 0;

    // Add initial walls
    std::pair<int, int> dirs[4] = {{2, 0}, {-2, 0}, {0, 2}, {0, -2}};
    for (auto [dr, dc] : dirs) {
      int nr = 1 + dr, nc = 1 + dc;
      if (inside(nr, nc)) {
        walls.emplace_back(nr, nc);
      }
    }

    while (!walls.empty()) {
      // Pick random wall
      int idx = irand(0, walls.size() - 1);
      auto [r, c] = walls[idx];
      walls[idx] = walls.back();
      walls.pop_back();

      if (at(r, c) == 1) {
        // Count adjacent passages
        int passage_count = 0;
        for (auto [dr, dc] : dirs) {
          int nr = r + dr, nc = c + dc;
          if (inside(nr, nc) && at(nr, nc) == 0) {
            passage_count++;
          }
        }

        // Only carve if exactly one adjacent passage
        if (passage_count == 1) {
          at(r, c) = 0;

          // Carve connecting cell
          for (auto [dr, dc] : dirs) {
            int nr = r + dr / 2, nc = c + dc / 2;
            if (inside(nr, nc) && at(nr, nc) == 0) {
              at(r - dr / 2, c - dc / 2) = 0;
              break;
            }
          }

          // Add new walls
          for (auto [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            if (inside(nr, nc) && at(nr, nc) == 1) {
              walls.emplace_back(nr, nc);
            }
          }
        }
      }
    }
  }

  // Generate maze with hybrid algorithm
  inline void generate_hybrid() noexcept {
    // Start with DFS
    generate_dfs();

    // Add some random openings for variety (10% of walls)
    int opening_count = (H * W) / 50;
    for (int i = 0; i < opening_count; ++i) {
      int r = irand(2, H - 3);
      int c = irand(2, W - 3);

      // Only open walls that connect two passages
      if (at(r, c) == 1) {
        int passage_neighbors = 0;
        if (at(r - 1, c) == 0) passage_neighbors++;
        if (at(r + 1, c) == 0) passage_neighbors++;
        if (at(r, c - 1) == 0) passage_neighbors++;
        if (at(r, c + 1) == 0) passage_neighbors++;

        if (passage_neighbors >= 2) {
          at(r, c) = 0;
        }
      }
    }
  }

  // Main generation function
  inline void generate(int h = 21, int w = 31,
                      MazeAlgorithm algo = MazeAlgorithm::HYBRID) {
    H = (h % 2 ? h : h + 1);
    W = (w % 2 ? w : w + 1);
    grid.assign(H * W, 1);
    algorithm = algo;

    // Set start and goal
    start = {1, 1};
    goal = {H - 2, W - 2};

    // Generate based on algorithm
    switch (algorithm) {
    case MazeAlgorithm::DFS_BACKTRACK:
      generate_dfs();
      break;
    case MazeAlgorithm::PRIMS:
      generate_prims();
      break;
    case MazeAlgorithm::HYBRID:
      generate_hybrid();
      break;
    }

    // Ensure start and goal are open
    at(start.first, start.second) = 0;
    at(goal.first, goal.second) = 0;
  }
};
