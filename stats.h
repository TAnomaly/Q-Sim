#pragma once
#include <deque>
#include <iostream>
#include <numeric>

// Performance statistics tracker
struct Stats {
  int success_count{0};
  int total_episodes{0};
  int last_steps{0};
  float last_reward{0.f};

  std::deque<int> recent_steps;
  std::deque<float> recent_rewards;
  static constexpr int window_size = 100;

  inline void record_episode(bool success, int steps, float total_reward) {
    total_episodes++;
    if (success)
      success_count++;

    last_steps = steps;
    last_reward = total_reward;

    recent_steps.push_back(steps);
    recent_rewards.push_back(total_reward);

    if (recent_steps.size() > window_size) {
      recent_steps.pop_front();
      recent_rewards.pop_front();
    }
  }

  inline void print_stats(float epsilon) const {
    if (total_episodes % 100 == 0 && total_episodes > 0) {
      float success_rate = 100.0f * success_count / 100.0f;
      float avg_steps = recent_steps.empty()
                            ? 0.f
                            : std::accumulate(recent_steps.begin(),
                                              recent_steps.end(), 0.f) /
                                  recent_steps.size();
      float avg_reward = recent_rewards.empty()
                             ? 0.f
                             : std::accumulate(recent_rewards.begin(),
                                               recent_rewards.end(), 0.f) /
                                   recent_rewards.size();

      std::cout << "Ep: " << total_episodes << " | Success: " << success_rate
                << "% | Avg Steps: " << (int)avg_steps
                << " | Avg Reward: " << avg_reward << " | Eps: " << epsilon
                << std::endl;
    }
  }

  inline void reset_window() {
    success_count = 0;
    if (total_episodes % 100 == 0) {
      success_count = 0;
    }
  }
};
