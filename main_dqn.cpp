#include "dqn_agent.h"
#include "env.h"
#include "maze.h"
#include "renderer.h"
#include "stats.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

struct CoverageMap {
  static constexpr float threshold = 0.02f;

  int H{0}, W{0};
  std::vector<float> data;
  std::vector<uint8_t> passable;
  int passable_total{1};
  int covered{0};

  void init(const Maze &mz) {
    H = mz.H;
    W = mz.W;
    data.assign(H * W, 0.0f);
    passable.assign(H * W, 0);
    passable_total = 0;
    covered = 0;

    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        int idx = r * W + c;
        if (mz.at(r, c) == 0) {
          passable[idx] = 1;
          passable_total++;
        }
      }
    }
    passable_total = std::max(passable_total, 1);
  }

  void reset_values() {
    std::fill(data.begin(), data.end(), 0.0f);
    covered = 0;
  }

  void mark(int r, int c, float amount = 0.08f) {
    if (r < 0 || r >= H || c < 0 || c >= W)
      return;
    int idx = r * W + c;
    if (!passable[idx])
      return;

    float &cell = data[idx];
    bool was_covered = cell > threshold;
    cell = std::min(1.0f, cell + amount);
    if (!was_covered && cell > threshold)
      covered++;
  }

  void fade(float factor = 0.995f) {
    covered = 0;
    for (size_t i = 0; i < data.size(); ++i) {
      if (!passable[i])
        continue;
      data[i] *= factor;
      if (data[i] > threshold)
        covered++;
    }
  }

  float ratio() const {
    return static_cast<float>(covered) /
           static_cast<float>(passable_total);
  }
};

void carve(Maze &mz, int r0, int c0, int r1, int c1) {
  if (r0 > r1)
    std::swap(r0, r1);
  if (c0 > c1)
    std::swap(c0, c1);

  r0 = std::clamp(r0, 0, mz.H - 1);
  r1 = std::clamp(r1, 0, mz.H - 1);
  c0 = std::clamp(c0, 0, mz.W - 1);
  c1 = std::clamp(c1, 0, mz.W - 1);

  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      mz.at(r, c) = 0;
    }
  }
}

void generateBuildingLayout(Maze &mz) {
  const int H = 11;
  const int W = 11;
  mz.H = H;
  mz.W = W;
  mz.grid.assign(H * W, 1);

  // Start & goal placed at opposite corners
  mz.start = {1, 1};
  mz.goal = {H - 2, W - 2};

  // Main corridors (simulate building hallways)
  carve(mz, 1, 1, 1, W - 2);
  carve(mz, 1, W - 2, H - 2, W - 2);
  carve(mz, 1, 1, H - 2, 1);
  carve(mz, 5, 1, 5, W - 2);
  carve(mz, 5, 1, H - 2, 1);
  carve(mz, 8, 1, 8, W - 3);
  carve(mz, 3, 3, 3, W - 3);
  carve(mz, 7, 3, 7, W - 4);

  // Rooms opening off corridors
  carve(mz, 2, 4, 2, 6);
  carve(mz, 4, 7, 4, 9);
  carve(mz, 6, 4, 6, 7);
  carve(mz, 8, 6, 8, 8);
  carve(mz, 9, 3, 9, 5);

  // Keep deliberate obstacles for navigation
  mz.at(4, 5) = 1;
  mz.at(6, 5) = 1;
  mz.at(8, 5) = 1;
  mz.at(7, 7) = 1;

  mz.at(mz.start.first, mz.start.second) = 0;
  mz.at(mz.goal.first, mz.goal.second) = 0;
}

void updateMappingView(bool show, const Maze &mz, const CoverageMap &coverage,
                       const PathTrail &trail, int agent_r, int agent_c,
                       float coverage_ratio, bool show_grid) {
  static const std::string window_name = "Drone Thermal Mapper";
  static bool window_created = false;

  if (!show) {
    if (window_created) {
      cv::destroyWindow(window_name);
      window_created = false;
    }
    return;
  }

  if (!window_created) {
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    window_created = true;
  }

  const int resolution = 720;
  cv::Mat canvas(resolution, resolution, CV_8UC3);
  canvas.setTo(cv::Scalar(18, 14, 28));

  int cell_w = std::max(1, resolution / mz.W);
  int cell_h = std::max(1, resolution / mz.H);
  int offset_x = (resolution - cell_w * mz.W) / 2;
  int offset_y = (resolution - cell_h * mz.H) / 2;

  const float depth_scale = 0.45f;
  const cv::Point2f depth_offset(-cell_w * depth_scale, -cell_h * depth_scale);

  for (int r = 0; r < mz.H; ++r) {
    for (int c = 0; c < mz.W; ++c) {
      cv::Point2f base_ll(offset_x + c * cell_w, offset_y + (r + 1) * cell_h);
      cv::Point2f base_lr = base_ll + cv::Point2f(static_cast<float>(cell_w), 0.f);
      cv::Point2f base_ul = base_ll - cv::Point2f(0.f, static_cast<float>(cell_h));
      cv::Point2f base_ur = base_ul + cv::Point2f(static_cast<float>(cell_w), 0.f);

      std::array<cv::Point2f, 4> base = {base_ll, base_lr, base_ur, base_ul};
      std::array<cv::Point2f, 4> top;
      for (int i = 0; i < 4; ++i)
        top[i] = base[i] + depth_offset;

      if (mz.at(r, c) == 1) {
        std::vector<cv::Point> base_poly, top_poly;
        base_poly.reserve(4);
        top_poly.reserve(4);
        for (int i = 0; i < 4; ++i) {
          base_poly.emplace_back(cv::Point(cvRound(base[i].x), cvRound(base[i].y)));
          top_poly.emplace_back(cv::Point(cvRound(top[i].x), cvRound(top[i].y)));
        }
        cv::fillConvexPoly(canvas, base_poly, cv::Scalar(35, 30, 45), cv::LINE_AA);
        cv::fillConvexPoly(canvas, top_poly, cv::Scalar(50, 45, 60), cv::LINE_AA);
        std::vector<cv::Point> side = {
            cv::Point(cvRound(base_ll.x), cvRound(base_ll.y)),
            cv::Point(cvRound(base_lr.x), cvRound(base_lr.y)),
            cv::Point(cvRound(top[1].x), cvRound(top[1].y)),
            cv::Point(cvRound(top[0].x), cvRound(top[0].y))};
        cv::fillConvexPoly(canvas, side, cv::Scalar(28, 24, 38), cv::LINE_AA);
        continue;
      }

      float intensity = 0.0f;
      int idx = r * mz.W + c;
      if (idx >= 0 && idx < static_cast<int>(coverage.data.size()))
        intensity = std::clamp(coverage.data[idx], 0.0f, 1.0f);

      float rr, gg, bb;
      thermalColor(intensity, rr, gg, bb);
      cv::Scalar top_col(bb * 255.0f, gg * 255.0f, rr * 255.0f);
      cv::Scalar side_col(bb * 110.0f, gg * 110.0f, rr * 110.0f);
      cv::Scalar front_col(bb * 150.0f, gg * 150.0f, rr * 150.0f);

      std::vector<cv::Point> top_poly;
      top_poly.reserve(4);
      for (auto &pt : top) {
        top_poly.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
      }
      cv::fillConvexPoly(canvas, top_poly, top_col, cv::LINE_AA);

      std::vector<cv::Point> front = {
          cv::Point(cvRound(base_ll.x), cvRound(base_ll.y)),
          cv::Point(cvRound(base_lr.x), cvRound(base_lr.y)),
          cv::Point(cvRound(top[1].x), cvRound(top[1].y)),
          cv::Point(cvRound(top[0].x), cvRound(top[0].y))};
      cv::fillConvexPoly(canvas, front, front_col, cv::LINE_AA);

      std::vector<cv::Point> side = {
          cv::Point(cvRound(base_lr.x), cvRound(base_lr.y)),
          cv::Point(cvRound(base_ur.x), cvRound(base_ur.y)),
          cv::Point(cvRound(top[2].x), cvRound(top[2].y)),
          cv::Point(cvRound(top[1].x), cvRound(top[1].y))};
      cv::fillConvexPoly(canvas, side, side_col, cv::LINE_AA);

      cv::polylines(canvas, top_poly, true,
                    cv::Scalar(255 * bb, 255 * gg, 255 * rr, 0.8f), 1,
                    cv::LINE_AA);
      std::vector<cv::Point> base_poly;
      for (auto &pt : base)
        base_poly.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
      cv::polylines(canvas, base_poly, true, cv::Scalar(20, 20, 30), 1,
                    cv::LINE_AA);
    }
  }

  if (show_grid) {
    cv::Scalar grid_col(50, 60, 90);
    for (int c = 0; c <= mz.W; ++c) {
      int x = offset_x + c * cell_w;
      cv::line(canvas, {x, offset_y}, {x, offset_y + mz.H * cell_h}, grid_col,
               1, cv::LINE_AA);
    }
    for (int r = 0; r <= mz.H; ++r) {
      int y = offset_y + r * cell_h;
      cv::line(canvas, {offset_x, y}, {offset_x + mz.W * cell_w, y}, grid_col,
               1, cv::LINE_AA);
    }
  }

  if (trail.points.size() >= 2) {
    std::vector<cv::Point> pts;
    pts.reserve(trail.points.size());
    for (const auto &p : trail.points) {
      int x = offset_x + p.c * cell_w + cell_w / 2;
      int y = offset_y + p.r * cell_h + cell_h / 2;
      pts.emplace_back(x, y);
    }
    cv::polylines(canvas, pts, false, cv::Scalar(180, 220, 255), 3,
                  cv::LINE_AA);
  }

  int agent_x = offset_x + agent_c * cell_w + cell_w / 2;
  int agent_y = offset_y + agent_r * cell_h + cell_h / 2;
  cv::circle(canvas, {agent_x, agent_y}, std::max(6, cell_w / 3),
             cv::Scalar(50, 255, 255), cv::FILLED, cv::LINE_AA);
  cv::circle(canvas, {agent_x, agent_y}, std::max(6, cell_w / 3),
             cv::Scalar(10, 70, 140), 2, cv::LINE_AA);

  std::ostringstream hud;
  hud << "Coverage " << std::fixed << std::setprecision(1)
      << coverage_ratio * 100.0f << "%";
  cv::putText(canvas, hud.str(), {24, 48}, cv::FONT_HERSHEY_DUPLEX, 0.9,
              cv::Scalar(240, 240, 255), 2, cv::LINE_AA);

  cv::imshow(window_name, canvas);
  cv::waitKey(1);
}

int main() {
  if (!glfwInit()) {
    std::cerr << "GLFW init fail\n";
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_SAMPLES, 4); // Anti-aliasing
  GLFWwindow *win =
      glfwCreateWindow(1400, 900, "DQN Maze Solver", nullptr, nullptr);
  if (!win) {
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);
  glEnable(GL_MULTISAMPLE);

  // Initialize environment and DQN agent
  Env env;
  generateBuildingLayout(env.mz);
  env.reset(false);

  std::cout << "\n=== DQN Building Mapper - Thermal View ===\n";
  std::cout << "Layout size: " << env.mz.H << "x" << env.mz.W << "\n";

  DQNAgent agent(env.mz.H, env.mz.W, 4);
  CoverageMap coverage;
  coverage.init(env.mz);
  coverage.mark(env.r, env.c, 0.4f);

  auto warmup_agent = [&](int episodes) {
    std::cout << "Calibrating drone agent (" << episodes << " episodes)...\n";
    for (int ep = 0; ep < episodes; ++ep) {
      env.reset(false);
      bool done = false;
      while (!done) {
        int old_r = env.r, old_c = env.c;
        int action = agent.act(env.mz, env.r, env.c);
        auto [nr, nc, reward, finished] = env.step(action);

        agent.remember(env.mz, old_r, old_c, action, reward, nr, nc, finished);
        agent.train_step();
        agent.decay_epsilon();
        done = finished;
      }
      if ((ep + 1) % 20 == 0) {
        glfwPollEvents();
        if (glfwWindowShouldClose(win))
          break;
        std::cout << "  Warmup progress: " << (ep + 1) << "/" << episodes
                  << std::endl;
      }
    }
    env.reset(false);
    coverage.init(env.mz);
    coverage.mark(env.r, env.c, 0.45f);
    agent.epsilon = std::max(agent.epsilon, 0.12f);
  };

  warmup_agent(150);

  // Visualization and stats
  PathTrail trail;
  Stats stats;
  bool paused = false;
  bool show_heatmap = false;
  bool show_trail = true;
  bool show_curves = true;
  bool show_mapping = true;
  bool show_grid = true;
  bool victory_pause = false;
  double victory_time = 0.0;
  int frameStep = 1;
  float total_reward = 0.f;
  float total_loss = 0.f;
  int train_steps = 0;

  std::cout << "\nControls:\n";
  std::cout << "  P     - Pause/Resume\n";
  std::cout << "  H     - Toggle Q-value Heatmap\n";
  std::cout << "  M     - Toggle Mapping Overlay\n";
  std::cout << "  G     - Toggle Grid Overlay\n";
  std::cout << "  T     - Toggle Path Trail\n";
  std::cout << "  C     - Toggle Training Curves\n";
  std::cout << "  R     - Regenerate Maze\n";
  std::cout << "  S     - Save Model\n";
  std::cout << "  L     - Load Model\n";
  std::cout << "  +/-   - Speed up/down (training steps per frame)\n";
  std::cout << "  ESC   - Exit\n\n";

  // Key press tracking
  bool key_p_pressed = false, key_h_pressed = false, key_m_pressed = false;
  bool key_g_pressed = false, key_t_pressed = false;
  bool key_c_pressed = false, key_r_pressed = false, key_s_pressed = false;
  bool key_l_pressed = false, key_plus_pressed = false,
       key_minus_pressed = false;

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    glfwMakeContextCurrent(win);

    // Handle key presses
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      break;

    if (glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS && !key_p_pressed) {
      paused = !paused;
      key_p_pressed = true;
      std::cout << (paused ? "Paused\n" : "Resumed\n");
    }
    if (glfwGetKey(win, GLFW_KEY_P) == GLFW_RELEASE)
      key_p_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS && !key_h_pressed) {
      show_heatmap = !show_heatmap;
      key_h_pressed = true;
      std::cout << (show_heatmap ? "Heatmap ON\n" : "Heatmap OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_H) == GLFW_RELEASE)
      key_h_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_M) == GLFW_PRESS && !key_m_pressed) {
      show_mapping = !show_mapping;
      if (show_mapping && show_heatmap) {
        show_heatmap = false;
        std::cout << "Heatmap OFF (mapping overlay active)\n";
      }
      key_m_pressed = true;
      std::cout << (show_mapping ? "Mapping overlay ON\n"
                                 : "Mapping overlay OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_M) == GLFW_RELEASE)
      key_m_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS && !key_g_pressed) {
      show_grid = !show_grid;
      key_g_pressed = true;
      std::cout << (show_grid ? "Grid overlay ON\n" : "Grid overlay OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_G) == GLFW_RELEASE)
      key_g_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS && !key_t_pressed) {
      show_trail = !show_trail;
      key_t_pressed = true;
      std::cout << (show_trail ? "Trail ON\n" : "Trail OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_T) == GLFW_RELEASE)
      key_t_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS && !key_c_pressed) {
      show_curves = !show_curves;
      key_c_pressed = true;
      std::cout << (show_curves ? "Training curves ON\n"
                                : "Training curves OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_C) == GLFW_RELEASE)
      key_c_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS && !key_r_pressed) {
      generateBuildingLayout(env.mz);
      env.reset(false);
      agent = DQNAgent(env.mz.H, env.mz.W, 4);
      warmup_agent(150);
      coverage.init(env.mz);
      coverage.mark(env.r, env.c, 0.45f);
      trail.clear();
      stats = Stats();
      key_r_pressed = true;
      std::cout << "Building layout reset & agent recalibrated\n";
    }
    if (glfwGetKey(win, GLFW_KEY_R) == GLFW_RELEASE)
      key_r_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS && !key_s_pressed) {
      if (agent.save("dqn_model.bin"))
        std::cout << "Model saved to dqn_model.bin\n";
      else
        std::cout << "Save failed\n";
      key_s_pressed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_RELEASE)
      key_s_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS && !key_l_pressed) {
      if (agent.load("dqn_model.bin"))
        std::cout << "Model loaded from dqn_model.bin\n";
      else
        std::cout << "Load failed\n";
      key_l_pressed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_L) == GLFW_RELEASE)
      key_l_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS && !key_plus_pressed) {
      frameStep = std::min(frameStep * 2, 512);
      key_plus_pressed = true;
      std::cout << "Speed: " << frameStep << "x\n";
    }
    if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_RELEASE)
      key_plus_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS && !key_minus_pressed) {
      frameStep = std::max(frameStep / 2, 1);
      key_minus_pressed = true;
      std::cout << "Speed: " << frameStep << "x\n";
    }
    if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_RELEASE)
      key_minus_pressed = false;

    // Handle victory pause
    if (victory_pause) {
      if (glfwGetTime() - victory_time > 2.0) {
        victory_pause = false;
        env.reset(false);
        trail.clear();
        total_reward = 0.f;
      }
    }

    // Training loop
    if (!paused && !victory_pause) {
      for (int it = 0; it < frameStep; ++it) {
        int old_r = env.r, old_c = env.c;
        int a = agent.act(env.mz, env.r, env.c);
        auto [nr, nc, rew, done] = env.step(a);

        // Store experience and train
        agent.remember(env.mz, old_r, old_c, a, rew, nr, nc, done);
        float loss = agent.train_step();
        float loss2 = agent.train_step();
        if (loss > 0.0f) {
          total_loss += loss;
          train_steps++;
        }
        if (loss2 > 0.0f) {
          total_loss += loss2;
          train_steps++;
        }

        total_reward += rew;
        coverage.fade(0.9995f);
        coverage.mark(old_r, old_c, 0.015f);
        coverage.mark(nr, nc, 0.12f);

        if (show_trail && frameStep <= 16) {
          trail.add(env.r, env.c, glfwGetTime());
        }

        if (done) {
          bool success = (nr == env.mz.goal.first && nc == env.mz.goal.second);

          if (success) {
            victory_pause = true;
            victory_time = glfwGetTime();
            std::cout << "\nVICTORY! Episode " << stats.total_episodes + 1
                      << " in " << env.steps << " steps, reward: " << total_reward
                      << "\n"
                      << std::endl;
          }

          float avg_loss = train_steps > 0 ? total_loss / train_steps : 0.0f;
          stats.record_episode(success, env.steps, total_reward);
          stats.record_loss(avg_loss);
          stats.print_stats(agent.epsilon, avg_loss);
          stats.reset_window();

          if (!victory_pause) {
            env.reset(false);
            coverage.mark(env.r, env.c, 0.25f);
            trail.clear();
            total_reward = 0.f;
          }

          total_loss = 0.f;
          train_steps = 0;
        }

        agent.decay_epsilon();
      }
    }

    // Window title
    float coverage_ratio = coverage.ratio();
    std::ostringstream title;
    if (victory_pause) {
      title << "VICTORY! | Ep: " << env.episode << " | Steps: " << env.steps
            << " | Reward: " << std::fixed << std::setprecision(1)
            << total_reward;
    } else {
      title << "DQN Drone Map | Ep: " << env.episode << " | Step: " << env.steps
            << " | Eps: " << std::fixed << std::setprecision(3) << agent.epsilon
            << " | Buffer: " << agent.replay_buffer.size << "/"
            << agent.replay_buffer.capacity << " | Speed: " << frameStep << "x"
            << " | Coverage: " << std::setprecision(1) << std::fixed
            << coverage_ratio * 100.0f << "%";
    }
    glfwSetWindowTitle(win, title.str().c_str());

    // Rendering
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.01f, 0.02f, 0.03f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw maze base (heatmap disabled when mapping active)
    bool heatmap_active = show_heatmap && !show_mapping;
    drawMaze(env.mz, heatmap_active ? &agent : nullptr, heatmap_active);

    if (show_mapping) {
      drawCoverage(env.mz, coverage.data);
      if (show_grid) {
        drawGrid(env.mz, 0.12f);
      }
    } else if (show_grid) {
      drawGrid(env.mz, 0.08f);
    }

    // Draw trail
    if (show_trail) {
      trail.draw(env.mz, glfwGetTime());
    }

    // Draw agent
    drawAgent(env.mz, env.r, env.c, agent.epsilon, victory_pause);

    // Draw training curves
    if (show_curves) {
      stats.draw_curves(w, h);
    }

    glDisable(GL_BLEND);
    glfwSwapBuffers(win);

    updateMappingView(show_mapping, env.mz, coverage, trail, env.r, env.c,
                      coverage_ratio, show_grid);
  }

  std::cout << "\nTraining completed!\n";
  std::cout << "Total episodes: " << stats.total_episodes << "\n";
  std::cout << "Success rate: "
            << (100.0f * stats.success_count /
                std::max(stats.total_episodes, 1))
            << "%\n";

  updateMappingView(false, env.mz, coverage, trail, env.r, env.c, 0.0f,
                    show_grid);
  cv::destroyAllWindows();
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
