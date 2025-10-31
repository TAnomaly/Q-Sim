#include "env.h"
#include "maze.h"
#include "qagent.h"
#include "renderer.h"
#include "stats.h"
#include <GLFW/glfw3.h>
#include <iomanip>
#include <iostream>
#include <sstream>

int main() {
  if (!glfwInit()) {
    std::cerr << "GLFW init fail\n";
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  GLFWwindow *win =
      glfwCreateWindow(1200, 800, "Q-Learning Maze", nullptr, nullptr);
  if (!win) {
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  // Initialize environment and agent
  Env env;
  env.mz.generate(21, 31);
  env.reset(false);
  QAgent agent(env.mz.H, env.mz.W);

  // Visualization and stats
  PathTrail trail;
  Stats stats;
  bool paused = false;
  bool show_heatmap = false;
  bool show_trail = true;
  bool victory_pause = false;
  double victory_time = 0.0;
  int frameStep = 1;
  float total_reward = 0.f;

  std::cout << "\n=== Q-Learning Maze Solver ===\n";
  std::cout << "Controls:\n";
  std::cout << "  P     - Pause/Resume\n";
  std::cout << "  H     - Toggle Q-value Heatmap\n";
  std::cout << "  T     - Toggle Path Trail\n";
  std::cout << "  R     - Regenerate Maze\n";
  std::cout << "  S     - Save Model\n";
  std::cout << "  L     - Load Model\n";
  std::cout << "  +/-   - Speed up/down\n";
  std::cout << "  ESC   - Exit\n\n";

  // Key press tracking
  bool key_p_pressed = false, key_h_pressed = false, key_t_pressed = false;
  bool key_r_pressed = false, key_s_pressed = false, key_l_pressed = false;
  bool key_plus_pressed = false, key_minus_pressed = false;

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();

    // Handle key presses with debounce
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      break;

    if (glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS && !key_p_pressed) {
      paused = !paused;
      key_p_pressed = true;
      std::cout << (paused ? "⏸ Paused\n" : "▶ Resumed\n");
    }
    if (glfwGetKey(win, GLFW_KEY_P) == GLFW_RELEASE)
      key_p_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS && !key_h_pressed) {
      show_heatmap = !show_heatmap;
      key_h_pressed = true;
      std::cout << (show_heatmap ? "🌡 Heatmap ON\n" : "🌡 Heatmap OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_H) == GLFW_RELEASE)
      key_h_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS && !key_t_pressed) {
      show_trail = !show_trail;
      key_t_pressed = true;
      std::cout << (show_trail ? "🛤 Trail ON\n" : "🛤 Trail OFF\n");
    }
    if (glfwGetKey(win, GLFW_KEY_T) == GLFW_RELEASE)
      key_t_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS && !key_r_pressed) {
      env.reset(true);
      agent = QAgent(env.mz.H, env.mz.W);
      trail.clear();
      stats = Stats();
      key_r_pressed = true;
      std::cout << "🔄 Maze regenerated\n";
    }
    if (glfwGetKey(win, GLFW_KEY_R) == GLFW_RELEASE)
      key_r_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS && !key_s_pressed) {
      if (agent.save("qtable.bin"))
        std::cout << "💾 Model saved to qtable.bin\n";
      else
        std::cout << "❌ Save failed\n";
      key_s_pressed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_RELEASE)
      key_s_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS && !key_l_pressed) {
      if (agent.load("qtable.bin"))
        std::cout << "📂 Model loaded from qtable.bin\n";
      else
        std::cout << "❌ Load failed\n";
      key_l_pressed = true;
    }
    if (glfwGetKey(win, GLFW_KEY_L) == GLFW_RELEASE)
      key_l_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS && !key_plus_pressed) {
      frameStep = std::min(frameStep * 2, 1024);
      key_plus_pressed = true;
      std::cout << "⚡ Speed: " << frameStep << "x\n";
    }
    if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_RELEASE)
      key_plus_pressed = false;

    if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS && !key_minus_pressed) {
      frameStep = std::max(frameStep / 2, 1);
      key_minus_pressed = true;
      std::cout << "🐌 Speed: " << frameStep << "x\n";
    }
    if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_RELEASE)
      key_minus_pressed = false;

    // Handle victory pause
    if (victory_pause) {
      if (glfwGetTime() - victory_time > 1.5) {
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
        int a = agent.act(env.r, env.c);
        auto [nr, nc, rew, done] = env.step(a);
        agent.update(old_r, old_c, a, rew, nr, nc, done);
        total_reward += rew;

        if (show_trail && frameStep <= 8) {
          trail.add(env.r, env.c, glfwGetTime());
        }

        if (done) {
          bool success = (nr == env.mz.goal.first && nc == env.mz.goal.second);

          if (success) {
            victory_pause = true;
            victory_time = glfwGetTime();
            std::cout << "\n🎯 VICTORY! Episode " << stats.total_episodes + 1
                      << " in " << env.steps << " steps, reward: " << total_reward
                      << "\n" << std::endl;
          }

          stats.record_episode(success, env.steps, total_reward);
          stats.print_stats(agent.eps);
          stats.reset_window();

          if (!victory_pause) {
            env.reset(false);
            trail.clear();
            total_reward = 0.f;
          }
        }

        agent.decay();
      }
    }

    // Window title
    std::ostringstream title;
    if (victory_pause) {
      title << "🎉 VICTORY! 🎉 | Ep: " << env.episode << " | Steps: " << env.steps
            << " | Speed: " << frameStep << "x";
    } else {
      title << "Q-Learning | Ep: " << env.episode << " | Step: " << env.steps
            << " | Eps: " << std::fixed << std::setprecision(3) << agent.eps
            << " | Speed: " << frameStep << "x";
      if (show_heatmap)
        title << " | 🌡";
      if (show_trail)
        title << " | 🛤";
    }
    glfwSetWindowTitle(win, title.str().c_str());

    // Rendering
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.02f, 0.03f, 0.05f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable blending for trail transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw maze with optional heatmap
    drawMaze(env.mz, show_heatmap ? &agent : nullptr, show_heatmap);

    // Draw trail
    if (show_trail) {
      trail.draw(env.mz, glfwGetTime());
    }

    // Draw agent
    drawAgent(env.mz, env.r, env.c, agent.eps, victory_pause);

    glDisable(GL_BLEND);
    glfwSwapBuffers(win);
  }

  std::cout << "\n✓ Total episodes: " << stats.total_episodes << "\n";
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
