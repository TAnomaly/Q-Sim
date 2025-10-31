#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h> // AVX2/AVX-512
#include <random>
#include <vector>

// Aligned allocator for SIMD operations
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template<typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;

  template<typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T *allocate(std::size_t n) {
    if (n == 0) return nullptr;

    std::size_t size = n * sizeof(T);

    // Use aligned_alloc (C11/C++17)
    void *ptr = std::aligned_alloc(Alignment, ((size + Alignment - 1) / Alignment) * Alignment);

    if (!ptr) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) noexcept {
    if (p) std::free(p);
  }

  template<typename U>
  bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
    return true;
  }

  template<typename U>
  bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
    return false;
  }
};

template <typename T, std::size_t A>
using AlignedVector = std::vector<T, AlignedAllocator<T, A>>;

// Activation functions with SIMD
namespace activation {

// ReLU with AVX2
inline void relu_avx2(float *data, int size) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  int i = 0;

  // Process 8 floats at a time
  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(data + i);
    __m256 result = _mm256_max_ps(x, zero);
    _mm256_storeu_ps(data + i, result);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    data[i] = std::max(0.0f, data[i]);
  }
}

// Leaky ReLU with AVX2
inline void leaky_relu_avx2(float *data, int size,
                             float alpha = 0.01f) noexcept {
  const __m256 alpha_vec = _mm256_set1_ps(alpha);
  const __m256 zero = _mm256_setzero_ps();
  int i = 0;

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(data + i);
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 neg = _mm256_mul_ps(x, alpha_vec);
    __m256 result = _mm256_blendv_ps(neg, x, mask);
    _mm256_storeu_ps(data + i, result);
  }

  for (; i < size; ++i) {
    data[i] = data[i] > 0.0f ? data[i] : alpha * data[i];
  }
}

} // namespace activation

// Dense layer with SIMD optimization
class DenseLayer {
public:
  int input_size;
  int output_size;

  std::vector<float> weights;
  std::vector<float> biases;
  std::vector<float> output;

  // For training
  std::vector<float> weight_gradients;
  std::vector<float> bias_gradients;
  std::vector<float> input_cache;

  DenseLayer(int in_size, int out_size)
      : input_size(in_size), output_size(out_size),
        weights(in_size * out_size, 0.0f), biases(out_size, 0.0f),
        output(out_size, 0.0f), weight_gradients(in_size * out_size, 0.0f),
        bias_gradients(out_size, 0.0f), input_cache(in_size, 0.0f) {

    // He initialization for ReLU
    std::random_device rd;
    std::mt19937 gen(rd());
    float std = std::sqrt(2.0f / in_size);
    std::normal_distribution<float> dist(0.0f, std);

    for (auto &w : weights) {
      w = dist(gen);
    }
  }

  // Forward pass with AVX2 optimization
  [[nodiscard]] inline const float *forward(const float *input) noexcept {
    // Cache input for backward pass
    std::memcpy(input_cache.data(), input, input_size * sizeof(float));

    // Matrix-vector multiplication: output = weights * input + bias
    for (int o = 0; o < output_size; ++o) {
      __m256 sum = _mm256_setzero_ps();
      int i = 0;

      // SIMD dot product (unaligned loads)
      for (; i + 7 < input_size; i += 8) {
        __m256 w = _mm256_loadu_ps(&weights[o * input_size + i]);
        __m256 inp = _mm256_loadu_ps(&input[i]);
        sum = _mm256_fmadd_ps(w, inp, sum);
      }

      // Horizontal sum of SIMD register
      float result = 0.0f;
      alignas(32) float temp[8];
      _mm256_store_ps(temp, sum);
      for (int j = 0; j < 8; ++j)
        result += temp[j];

      // Handle remaining elements
      for (; i < input_size; ++i) {
        result += weights[o * input_size + i] * input[i];
      }

      output[o] = result + biases[o];
    }

    return output.data();
  }

  // Apply activation in-place
  inline void apply_relu() noexcept {
    activation::relu_avx2(output.data(), output_size);
  }

  inline void apply_leaky_relu(float alpha = 0.01f) noexcept {
    activation::leaky_relu_avx2(output.data(), output_size, alpha);
  }

  // Copy weights from another layer (for target network)
  inline void copy_weights_from(const DenseLayer &other) noexcept {
    std::memcpy(weights.data(), other.weights.data(),
                weights.size() * sizeof(float));
    std::memcpy(biases.data(), other.biases.data(),
                biases.size() * sizeof(float));
  }

  // Soft update: this = tau * other + (1 - tau) * this
  inline void soft_update_from(const DenseLayer &other, float tau) noexcept {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = tau * other.weights[i] + (1.0f - tau) * weights[i];
    }
    for (size_t i = 0; i < biases.size(); ++i) {
      biases[i] = tau * other.biases[i] + (1.0f - tau) * biases[i];
    }
  }
};

// Simple feed-forward neural network for DQN
class NeuralNetwork {
public:
  std::vector<DenseLayer> layers;
  std::vector<float> input_buffer;

  NeuralNetwork(const std::vector<int> &layer_sizes) {
    input_buffer.resize(layer_sizes[0]);

    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
      layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
    }
  }

  // Forward pass through all layers
  [[nodiscard]] inline const float *forward(const float *input) noexcept {
    const float *current = input;

    for (size_t i = 0; i < layers.size(); ++i) {
      current = layers[i].forward(current);

      // Apply activation (no activation on output layer)
      if (i < layers.size() - 1) {
        layers[i].apply_leaky_relu(0.01f);
      }
    }

    return current;
  }

  // Get output for a specific state
  [[nodiscard]] inline const float *
  predict(const float *state_input) noexcept {
    return forward(state_input);
  }

  // Copy entire network (for target network)
  inline void copy_from(const NeuralNetwork &other) noexcept {
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i].copy_weights_from(other.layers[i]);
    }
  }

  // Soft update from another network
  inline void soft_update_from(const NeuralNetwork &other, float tau) noexcept {
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i].soft_update_from(other.layers[i], tau);
    }
  }

  // Get number of parameters
  [[nodiscard]] size_t num_parameters() const noexcept {
    size_t total = 0;
    for (const auto &layer : layers) {
      total += layer.weights.size() + layer.biases.size();
    }
    return total;
  }
};

// Simple gradient descent optimizer
class SGDOptimizer {
public:
  float learning_rate;
  float momentum;
  float weight_decay;

  std::vector<std::vector<float>> weight_velocity;
  std::vector<std::vector<float>> bias_velocity;

  SGDOptimizer(float lr = 0.001f, float mom = 0.9f, float wd = 1e-4f)
      : learning_rate(lr), momentum(mom), weight_decay(wd) {}

  void initialize(const NeuralNetwork &net) {
    weight_velocity.clear();
    bias_velocity.clear();

    for (const auto &layer : net.layers) {
      weight_velocity.emplace_back(layer.weights.size(), 0.0f);
      bias_velocity.emplace_back(layer.biases.size(), 0.0f);
    }
  }

  // Simple gradient update with momentum
  void step(NeuralNetwork &net) noexcept {
    for (size_t l = 0; l < net.layers.size(); ++l) {
      auto &layer = net.layers[l];

      // Update weights
      for (size_t i = 0; i < layer.weights.size(); ++i) {
        float grad = layer.weight_gradients[i] + weight_decay * layer.weights[i];
        weight_velocity[l][i] = momentum * weight_velocity[l][i] - learning_rate * grad;
        layer.weights[i] += weight_velocity[l][i];
      }

      // Update biases
      for (size_t i = 0; i < layer.biases.size(); ++i) {
        float grad = layer.bias_gradients[i];
        bias_velocity[l][i] = momentum * bias_velocity[l][i] - learning_rate * grad;
        layer.biases[i] += bias_velocity[l][i];
      }

      // Zero gradients
      std::fill(layer.weight_gradients.begin(), layer.weight_gradients.end(), 0.0f);
      std::fill(layer.bias_gradients.begin(), layer.bias_gradients.end(), 0.0f);
    }
  }
};
