#pragma once
#include <GLFW/glfw3.h>
#include <algorithm>
#include <deque>
#include <iostream>
#include <numeric>
#include <vector>

// Performance statistics tracker with visualization
struct Stats {
  int success_count{0};
  int window_success_count{0};
  int total_episodes{0};
  int last_steps{0};
  float last_reward{0.f};
  float best_reward{-1000.0f};

  std::deque<int> recent_steps;
  std::deque<float> recent_rewards;
  std::deque<float> recent_success_rates;
  std::deque<float> recent_losses;

  static constexpr int window_size = 100;
  static constexpr int chart_history = 200;

  inline void record_episode(bool success, int steps, float total_reward) {
    total_episodes++;
    if (success) {
      success_count++;
      window_success_count++;
    }

    last_steps = steps;
    last_reward = total_reward;
    best_reward = std::max(best_reward, total_reward);

    recent_steps.push_back(steps);
    recent_rewards.push_back(total_reward);

    if (recent_steps.size() > chart_history) {
      recent_steps.pop_front();
      recent_rewards.pop_front();
    }

    // Track success rate
    if (total_episodes % 10 == 0) {
      float rate = 100.0f * window_success_count / 10.0f;
      recent_success_rates.push_back(rate);
      window_success_count = 0;

      if (recent_success_rates.size() > chart_history) {
        recent_success_rates.pop_front();
      }
    }
  }

  inline void record_loss(float loss) {
    recent_losses.push_back(loss);
    if (recent_losses.size() > chart_history) {
      recent_losses.pop_front();
    }
  }

  inline void print_stats(float epsilon, float loss = 0.0f) const {
    if (total_episodes % 50 == 0 && total_episodes > 0) {
      float success_rate =
          100.0f * success_count / std::min(total_episodes, 100);
      float avg_steps =
          recent_steps.empty()
              ? 0.f
              : std::accumulate(recent_steps.begin(), recent_steps.end(), 0.f) /
                    recent_steps.size();
      float avg_reward = recent_rewards.empty()
                             ? 0.f
                             : std::accumulate(recent_rewards.begin(),
                                               recent_rewards.end(), 0.f) /
                                   recent_rewards.size();

      std::cout << "Ep: " << total_episodes << " | Success: " << success_rate
                << "% | Avg Steps: " << (int)avg_steps
                << " | Avg Reward: " << avg_reward << " | Best: " << best_reward
                << " | Loss: " << loss << " | Eps: " << epsilon << std::endl;
    }
  }

  inline void reset_window() {
    if (total_episodes % 100 == 0) {
      success_count = 0;
    }
  }

  // Draw training curves on screen
  inline void draw_curves(int win_w, int win_h) const {
    if (recent_success_rates.empty() && recent_rewards.empty())
      return;

    // Chart area (bottom right corner)
    float chart_w = 300.0f;
    float chart_h = 150.0f;
    float margin = 10.0f;
    float chart_x = win_w - chart_w - margin;
    float chart_y = margin;

    // Convert to OpenGL coordinates
    float gl_x = (chart_x / win_w) * 2.0f - 1.0f;
    float gl_y = 1.0f - (chart_y / win_h) * 2.0f;
    float gl_w = (chart_w / win_w) * 2.0f;
    float gl_h = (chart_h / win_h) * 2.0f;

    // Draw background
    glColor4f(0.05f, 0.05f, 0.08f, 0.85f);
    glBegin(GL_QUADS);
    glVertex2f(gl_x, gl_y);
    glVertex2f(gl_x + gl_w, gl_y);
    glVertex2f(gl_x + gl_w, gl_y - gl_h);
    glVertex2f(gl_x, gl_y - gl_h);
    glEnd();

    // Draw border
    glColor4f(0.3f, 0.3f, 0.4f, 0.8f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(gl_x, gl_y);
    glVertex2f(gl_x + gl_w, gl_y);
    glVertex2f(gl_x + gl_w, gl_y - gl_h);
    glVertex2f(gl_x, gl_y - gl_h);
    glEnd();

    // Draw success rate curve (green)
    if (!recent_success_rates.empty()) {
      glColor4f(0.2f, 0.8f, 0.3f, 0.9f);
      glLineWidth(2.5f);
      glBegin(GL_LINE_STRIP);

      for (size_t i = 0; i < recent_success_rates.size(); ++i) {
        float x = gl_x + (static_cast<float>(i) / chart_history) * gl_w;
        float y_norm = recent_success_rates[i] / 100.0f; // 0-100% -> 0-1
        float y = gl_y - gl_h + y_norm * gl_h;
        glVertex2f(x, y);
      }
      glEnd();
    }

    // Draw reward curve (cyan) - normalized to 0-1 range
    if (!recent_rewards.empty()) {
      float min_r = *std::min_element(recent_rewards.begin(), recent_rewards.end());
      float max_r = *std::max_element(recent_rewards.begin(), recent_rewards.end());
      float range = max_r - min_r;

      if (range > 0.01f) {
        glColor4f(0.3f, 0.7f, 1.0f, 0.8f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);

        for (size_t i = 0; i < recent_rewards.size(); ++i) {
          float x = gl_x + (static_cast<float>(i) / chart_history) * gl_w;
          float y_norm = (recent_rewards[i] - min_r) / range;
          float y = gl_y - gl_h + y_norm * gl_h;
          glVertex2f(x, y);
        }
        glEnd();
      }
    }

    // Draw reference line at 50%
    glColor4f(0.5f, 0.5f, 0.5f, 0.4f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    float mid_y = gl_y - gl_h * 0.5f;
    glVertex2f(gl_x, mid_y);
    glVertex2f(gl_x + gl_w, mid_y);
    glEnd();
  }
};
