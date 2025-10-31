#pragma once
#include "maze.h"
#include "neural_net.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// Experience for replay buffer
struct Experience {
  std::vector<float> state;
  int action;
  float reward;
  std::vector<float> next_state;
  bool done;

  Experience(const float *s, int a, float r, const float *ns, bool d, int state_size)
      : state(s, s + state_size), action(a), reward(r),
        next_state(ns, ns + state_size), done(d) {}
};

// Prioritized Experience Replay Buffer
class ReplayBuffer {
public:
  size_t capacity;
  size_t size{0};
  size_t position{0};
  std::vector<Experience> buffer;

  ReplayBuffer(size_t cap) : capacity(cap) { buffer.reserve(cap); }

  inline void add(const float *state, int action, float reward,
                  const float *next_state, bool done, int state_size) {
    if (buffer.size() < capacity) {
      buffer.emplace_back(state, action, reward, next_state, done, state_size);
      size++;
    } else {
      buffer[position] = Experience(state, action, reward, next_state, done, state_size);
    }
    position = (position + 1) % capacity;
  }

  [[nodiscard]] inline bool can_sample(size_t batch_size) const noexcept {
    return size >= batch_size;
  }

  // Sample random batch
  [[nodiscard]] std::vector<size_t> sample_indices(size_t batch_size) {
    std::vector<size_t> indices(batch_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, size - 1);

    for (auto &idx : indices) {
      idx = dist(gen);
    }
    return indices;
  }

  inline void clear() noexcept {
    buffer.clear();
    size = 0;
    position = 0;
  }
};

// DQN Agent with target network and Double DQN
class DQNAgent {
public:
  int state_size;
  int action_size;
  int H, W;

  // Networks
  NeuralNetwork policy_net;
  NeuralNetwork target_net;
  SGDOptimizer optimizer;

  // Replay buffer
  ReplayBuffer replay_buffer;

  // Hyperparameters
  static constexpr float gamma = 0.99f;
  static constexpr float tau = 0.015f; // Soft update rate
  static constexpr float eps_start = 1.0f;
  static constexpr float eps_end = 0.02f;
  static constexpr float eps_decay = 0.9995f;
  static constexpr size_t batch_size = 64;
  static constexpr size_t buffer_capacity = 50000;
  static constexpr int target_update_freq = 250; // Hard sync frequency

  float epsilon{eps_start};
  int steps{0};
  bool use_double_dqn{true};

  // Buffers for state encoding
  std::vector<float> state_buffer;
  std::vector<float> next_state_buffer;

  // Training batch buffers
  std::vector<float> batch_states;
  std::vector<float> batch_next_states;
  std::vector<int> batch_actions;
  std::vector<float> batch_rewards;
  std::vector<bool> batch_dones;
  std::vector<std::vector<float>> layer_deltas;

  DQNAgent(int h, int w, int actions = 4)
      : state_size(h * w + 2),
        action_size(actions),
        H(h), W(w),
        policy_net({state_size, 256, 256, 128, actions}),
        target_net({state_size, 256, 256, 128, actions}),
        optimizer(0.0004f, 0.9f, 5e-5f),
        replay_buffer(buffer_capacity) {

    // Resize buffers
    state_buffer.resize(state_size);
    next_state_buffer.resize(state_size);
    batch_states.resize(batch_size * state_size);
    batch_next_states.resize(batch_size * state_size);
    batch_actions.resize(batch_size);
    batch_rewards.resize(batch_size);
    batch_dones.resize(batch_size);

    layer_deltas.resize(policy_net.layers.size());
    for (size_t i = 0; i < layer_deltas.size(); ++i) {
      layer_deltas[i].assign(policy_net.layers[i].output_size, 0.0f);
    }

    // Initialize target network with same weights
    target_net.copy_from(policy_net);
    optimizer.initialize(policy_net);
  }

  // Encode state: maze layout + agent position (one-hot encoding)
  inline void encode_state(const Maze &mz, int r, int c,
                           float *output) noexcept {
    int idx = 0;

    // Encode maze (flattened)
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        output[idx++] = mz.at(i, j) == 1 ? 1.0f : 0.0f;
      }
    }

    // Encode agent position (normalized)
    output[idx++] = static_cast<float>(r) / H;
    output[idx++] = static_cast<float>(c) / W;
  }

  // Select action using epsilon-greedy
  [[nodiscard]] inline int act(const Maze &mz, int r, int c) noexcept {
    if (frand() < epsilon) {
      return irand(0, action_size - 1);
    }

    // Encode state and predict Q-values
    encode_state(mz, r, c, state_buffer.data());
    const float *q_values = policy_net.predict(state_buffer.data());

    // Find best action
    return std::distance(q_values,
                        std::max_element(q_values, q_values + action_size));
  }

  // Get best action (no exploration) - non-const for simplicity
  [[nodiscard]] inline int best_action(const Maze &mz, int r, int c) {
    encode_state(mz, r, c, state_buffer.data());
    const float *q_values = policy_net.predict(state_buffer.data());
    return std::distance(q_values,
                        std::max_element(q_values, q_values + action_size));
  }

  // Get Q-values for visualization
  [[nodiscard]] inline const float *get_q_values(const Maze &mz, int r, int c) {
    encode_state(mz, r, c, state_buffer.data());
    return policy_net.predict(state_buffer.data());
  }

  // Get max Q-value for a state
  [[nodiscard]] inline float max_q(const Maze &mz, int r, int c) {
    const float *q_vals = get_q_values(mz, r, c);
    return *std::max_element(q_vals, q_vals + action_size);
  }

  // Store experience in replay buffer
  inline void remember(const Maze &mz, int r, int c, int action, float reward,
                       int nr, int nc, bool done) noexcept {
    encode_state(mz, r, c, state_buffer.data());
    encode_state(mz, nr, nc, next_state_buffer.data());
    replay_buffer.add(state_buffer.data(), action, reward,
                     next_state_buffer.data(), done, state_size);
  }

  // Train on a batch from replay buffer
  inline float train_step() noexcept {
    if (!replay_buffer.can_sample(batch_size)) {
      return 0.0f;
    }

    // Sample batch
    auto indices = replay_buffer.sample_indices(batch_size);

    // Prepare batch
    for (size_t i = 0; i < batch_size; ++i) {
      const auto &exp = replay_buffer.buffer[indices[i]];
      std::memcpy(&batch_states[i * state_size], exp.state.data(),
                  state_size * sizeof(float));
      std::memcpy(&batch_next_states[i * state_size], exp.next_state.data(),
                  state_size * sizeof(float));
      batch_actions[i] = exp.action;
      batch_rewards[i] = exp.reward;
      batch_dones[i] = exp.done;
    }

    // Compute TD targets
    float total_loss = 0.0f;

    for (size_t i = 0; i < batch_size; ++i) {
      // Compute target Q-value
      float target = batch_rewards[i];
      if (!batch_dones[i]) {
        if (use_double_dqn) {
          const float *q_next_policy =
              policy_net.predict(&batch_next_states[i * state_size]);
          int best_action =
              std::distance(q_next_policy,
                            std::max_element(q_next_policy,
                                             q_next_policy + action_size));

          const float *q_next_target =
              target_net.predict(&batch_next_states[i * state_size]);
          target += gamma * q_next_target[best_action];
        } else {
          const float *q_next =
              target_net.predict(&batch_next_states[i * state_size]);
          float max_q_next =
              *std::max_element(q_next, q_next + action_size);
          target += gamma * max_q_next;
        }
      }

      // Forward pass for current state (keeps caches for gradients)
      const float *q_current =
          policy_net.predict(&batch_states[i * state_size]);
      float q_value = q_current[batch_actions[i]];

      // TD error and loss
      float td_error = target - q_value;
      total_loss += td_error * td_error;

      float grad_scale = -2.0f * td_error / static_cast<float>(batch_size);
      size_t last_layer_index = policy_net.layers.size() - 1;

      // Prepare delta for output layer
      auto &output_delta = layer_deltas[last_layer_index];
      std::fill(output_delta.begin(), output_delta.end(), 0.0f);
      int action = batch_actions[i];
      output_delta[action] = grad_scale;

      // Accumulate gradients for output layer parameters
      auto &output_layer = policy_net.layers.back();
      output_layer.bias_gradients[action] += grad_scale;
      const float *prev_activation = output_layer.input_cache.data();
      for (int j = 0; j < output_layer.input_size; ++j) {
        output_layer.weight_gradients[action * output_layer.input_size + j] +=
            grad_scale * prev_activation[j];
      }

      // Backpropagate through hidden layers
      for (int l = static_cast<int>(last_layer_index) - 1; l >= 0; --l) {
        auto &layer = policy_net.layers[l];
        auto &next_layer = policy_net.layers[l + 1];
        auto &delta_curr = layer_deltas[l];
        auto &delta_next = layer_deltas[l + 1];

        std::fill(delta_curr.begin(), delta_curr.end(), 0.0f);

        for (int neuron = 0; neuron < layer.output_size; ++neuron) {
          float sum = 0.0f;
          for (int k = 0; k < next_layer.output_size; ++k) {
            sum += next_layer.weights[k * next_layer.input_size + neuron] *
                   delta_next[k];
          }

          float activation_deriv =
              layer.output[neuron] > 0.0f ? 1.0f : 0.01f;
          float grad_hidden = sum * activation_deriv;

          delta_curr[neuron] = grad_hidden;
          layer.bias_gradients[neuron] += grad_hidden;

          const float *layer_input = layer.input_cache.data();
          for (int m = 0; m < layer.input_size; ++m) {
            layer.weight_gradients[neuron * layer.input_size + m] +=
                grad_hidden * layer_input[m];
          }
        }
      }
    }

    // Update network
    optimizer.step(policy_net);

    // Target network maintenance (mix of soft & periodic hard sync)
    if (steps % target_update_freq == 0) {
      target_net.copy_from(policy_net);
    } else {
      target_net.soft_update_from(policy_net, tau);
    }

    steps++;
    return total_loss / static_cast<float>(batch_size);
  }

  // Decay epsilon
  inline void decay_epsilon() noexcept {
    epsilon = std::max(eps_end, epsilon * eps_decay);
  }

  // Save model
  bool save(const char *filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
      return false;

    // Save metadata
    out.write(reinterpret_cast<const char *>(&H), sizeof(H));
    out.write(reinterpret_cast<const char *>(&W), sizeof(W));
    out.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    out.write(reinterpret_cast<const char *>(&steps), sizeof(steps));

    // Save policy network weights
    for (const auto &layer : policy_net.layers) {
      out.write(reinterpret_cast<const char *>(layer.weights.data()),
                layer.weights.size() * sizeof(float));
      out.write(reinterpret_cast<const char *>(layer.biases.data()),
                layer.biases.size() * sizeof(float));
    }

    return out.good();
  }

  // Load model
  bool load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
      return false;

    int h, w;
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    in.read(reinterpret_cast<char *>(&w), sizeof(w));

    if (h != H || w != W)
      return false;

    in.read(reinterpret_cast<char *>(&epsilon), sizeof(epsilon));
    in.read(reinterpret_cast<char *>(&steps), sizeof(steps));

    // Load policy network weights
    for (auto &layer : policy_net.layers) {
      in.read(reinterpret_cast<char *>(layer.weights.data()),
              layer.weights.size() * sizeof(float));
      in.read(reinterpret_cast<char *>(layer.biases.data()),
              layer.biases.size() * sizeof(float));
    }

    // Sync target network
    target_net.copy_from(policy_net);

    return in.good();
  }
};
