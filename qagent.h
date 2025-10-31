#pragma once
#include "maze.h"
#include <algorithm>
#include <fstream>
#include <vector>

// Q-Learning Agent - cache-friendly layout
struct QAgent {
  int H, W;
  static constexpr int A = 4;
  std::vector<float> Q;

  // Hyperparameters as constexpr for optimization
  static constexpr float alpha = 0.3f;
  static constexpr float gamma = 0.99f;
  static constexpr float eps_end = 0.01f;
  static constexpr float eps_decay = 0.99998f;

  float eps{1.0f};

  QAgent(int h, int w) : H(h), W(w), Q(h * w * A, 0.f) {}

  // Fast index calculation - inline and constexpr
  [[nodiscard]] constexpr inline int idx(int r, int c, int a) const noexcept {
    return (r * W + c) * A + a;
  }

  // Action selection - epsilon-greedy
  [[nodiscard]] inline int act(int r, int c) noexcept {
    if (frand() < eps)
      return irand(0, A - 1);

    // Manual unroll for best action search
    int base = idx(r, c, 0);
    float q0 = Q[base], q1 = Q[base + 1], q2 = Q[base + 2], q3 = Q[base + 3];

    int best = 0;
    float best_q = q0;

    if (q1 > best_q) {
      best = 1;
      best_q = q1;
    }
    if (q2 > best_q) {
      best = 2;
      best_q = q2;
    }
    if (q3 > best_q) {
      best = 3;
    }

    return best;
  }

  // Q-learning update - hot path
  inline void update(int r, int c, int a, float reward, int r1, int c1,
                     bool done) noexcept {
    float best_next = 0.f;

    if (!done) {
      int base = idx(r1, c1, 0);
      best_next = std::max({Q[base], Q[base + 1], Q[base + 2], Q[base + 3]});
    }

    float td_target = reward + gamma * best_next;
    float &q = Q[idx(r, c, a)];
    q += alpha * (td_target - q);
  }

  inline void decay() noexcept { eps = std::max(eps_end, eps * eps_decay); }

  // Get best action without exploration (for visualization)
  [[nodiscard]] inline int best_action(int r, int c) const noexcept {
    int base = idx(r, c, 0);
    float q0 = Q[base], q1 = Q[base + 1], q2 = Q[base + 2], q3 = Q[base + 3];

    int best = 0;
    float best_q = q0;

    if (q1 > best_q) {
      best = 1;
      best_q = q1;
    }
    if (q2 > best_q) {
      best = 2;
      best_q = q2;
    }
    if (q3 > best_q) {
      best = 3;
    }

    return best;
  }

  // Get max Q-value for a state
  [[nodiscard]] inline float max_q(int r, int c) const noexcept {
    int base = idx(r, c, 0);
    return std::max({Q[base], Q[base + 1], Q[base + 2], Q[base + 3]});
  }

  // Save Q-table to binary file
  bool save(const char *filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      return false;

    out.write(reinterpret_cast<const char *>(&H), sizeof(H));
    out.write(reinterpret_cast<const char *>(&W), sizeof(W));
    out.write(reinterpret_cast<const char *>(&eps), sizeof(eps));
    out.write(reinterpret_cast<const char *>(Q.data()),
              Q.size() * sizeof(float));

    return out.good();
  }

  // Load Q-table from binary file
  bool load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      return false;

    int h, w;
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    in.read(reinterpret_cast<char *>(&w), sizeof(w));

    if (h != H || w != W)
      return false;

    in.read(reinterpret_cast<char *>(&eps), sizeof(eps));
    in.read(reinterpret_cast<char *>(Q.data()), Q.size() * sizeof(float));

    return in.good();
  }
};
