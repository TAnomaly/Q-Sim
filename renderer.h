#pragma once
#include "dqn_agent.h"
#include "env.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

// Path trail for visualization
struct PathTrail {
  struct Point {
    int r, c;
    double time;
  };

  std::deque<Point> points;
  static constexpr int max_points = 100;
  static constexpr double fade_time = 1.0;

  inline void add(int r, int c, double time) {
    points.push_back({r, c, time});
    if (points.size() > max_points)
      points.pop_front();
  }

  inline void clear() { points.clear(); }

  inline void draw(const Maze &mz, double current_time) const {
    if (points.size() >= 2) {
      float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
      glLineWidth(3.0f);
      glColor4f(0.2f, 0.8f, 0.95f, 0.85f);
      glBegin(GL_LINE_STRIP);
      for (const auto &p : points) {
        float cx = -1.f + p.c * cw + cw * 0.5f;
        float cy = 1.f - p.r * ch - ch * 0.5f;
        glVertex2f(cx, cy);
      }
      glEnd();
    }

    float cw = 2.0f / mz.W, ch = 2.0f / mz.H;

    for (size_t i = 0; i < points.size(); ++i) {
      const auto &p = points[i];
      double age = current_time - p.time;
      if (age > fade_time)
        continue;

      // Smooth fade with easing
      float t = age / fade_time;
      float alpha = (1.0f - t) * (1.0f - t); // Quadratic easing

      // Color gradient along trail (blue to cyan)
      float progress = static_cast<float>(i) / points.size();
      float r = 0.1f + progress * 0.3f;
      float g = 0.4f + progress * 0.4f;
      float b = 0.8f + progress * 0.2f;

      float L = -1.f + p.c * cw, R = L + cw, T = 1.f - p.r * ch, B = T - ch;
      float pad = 0.25f;
      float l = L + pad * cw, r_pos = R - pad * cw, t_pos = T - pad * ch,
            b_pos = B + pad * ch;

      glColor4f(r, g, b, alpha * 0.6f);
      glBegin(GL_QUADS);
      glVertex2f(l, b_pos);
      glVertex2f(r_pos, b_pos);
      glVertex2f(r_pos, t_pos);
      glVertex2f(l, t_pos);
      glEnd();

      // Glow effect
      glColor4f(r, g, b, alpha * 0.2f);
      float glow_pad = 0.15f;
      float l2 = L + glow_pad * cw, r2 = R - glow_pad * cw,
            t2 = T - glow_pad * ch, b2 = B + glow_pad * ch;
      glBegin(GL_QUADS);
      glVertex2f(l2, b2);
      glVertex2f(r2, b2);
      glVertex2f(r2, t2);
      glVertex2f(l2, t2);
      glEnd();
    }
  }
};

inline void drawTrailLine(const Maze &mz, const PathTrail &trail,
                          double current_time, float alpha = 0.9f) {
  if (trail.points.size() < 2)
    return;

  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
  glLineWidth(4.0f);

  glBegin(GL_LINE_STRIP);
  for (size_t i = 0; i < trail.points.size(); ++i) {
    const auto &p = trail.points[i];
    double age = current_time - p.time;
    float age_norm =
        static_cast<float>(age / PathTrail::fade_time);
    float time_fade = std::clamp(1.0f - age_norm, 0.0f, 1.0f);
    float progress =
        static_cast<float>(i) / std::max<size_t>(trail.points.size() - 1, 1);

    float r = 0.15f + 0.55f * progress;
    float g = 0.45f + 0.35f * (1.0f - progress);
    float b = 0.85f + 0.15f * progress;
    float a = alpha * time_fade * (0.4f + 0.6f * progress);

    float cx = -1.f + p.c * cw + cw * 0.5f;
    float cy = 1.f - p.r * ch - ch * 0.5f;
    glColor4f(r, g, b, a);
    glVertex2f(cx, cy);
  }
  glEnd();
}

// Color utilities
inline void hsv_to_rgb(float h, float s, float v, float &r, float &g,
                       float &b) {
  int i = (int)(h * 6);
  float f = h * 6 - i;
  float p = v * (1 - s);
  float q = v * (1 - f * s);
  float t = v * (1 - (1 - f) * s);

  switch (i % 6) {
  case 0:
    r = v, g = t, b = p;
    break;
  case 1:
    r = q, g = v, b = p;
    break;
  case 2:
    r = p, g = v, b = t;
    break;
  case 3:
    r = p, g = q, b = v;
    break;
  case 4:
    r = t, g = p, b = v;
    break;
  case 5:
    r = v, g = p, b = q;
    break;
  }
}

inline void drawGrid(const Maze &mz, float alpha = 0.15f) {
  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
  glLineWidth(1.0f);
  glColor4f(1.f, 1.f, 1.f, alpha);
  glBegin(GL_LINES);
  for (int c = 0; c <= mz.W; ++c) {
    float x = -1.f + c * cw;
    glVertex2f(x, -1.f);
    glVertex2f(x, 1.f);
  }
  for (int r = 0; r <= mz.H; ++r) {
    float y = 1.f - r * ch;
    glVertex2f(-1.f, y);
    glVertex2f(1.f, y);
  }
  glEnd();
}

inline void thermalColor(float intensity, float &r, float &g, float &b) {
  float t = std::clamp(intensity, 0.0f, 1.0f);
  if (t < 0.25f) {
    float k = t / 0.25f;
    r = 0.0f;
    g = 0.2f + 0.6f * k;
    b = 0.6f + 0.4f * k;
  } else if (t < 0.5f) {
    float k = (t - 0.25f) / 0.25f;
    r = 0.2f * k;
    g = 0.8f + 0.15f * k;
    b = 1.0f - 0.7f * k;
  } else if (t < 0.75f) {
    float k = (t - 0.5f) / 0.25f;
    r = 0.2f + 0.5f * k;
    g = 0.95f - 0.45f * k;
    b = 0.3f * (1.0f - k);
  } else {
    float k = (t - 0.75f) / 0.25f;
    r = 0.7f + 0.3f * k;
    g = 0.5f - 0.4f * k;
    b = 0.0f + 0.05f * (1.0f - k);
  }
}

inline void drawThermalPrism(float L, float B, float R, float T,
                             float intensity) {
  float r_base, g_base, b_base;
  thermalColor(intensity, r_base, g_base, b_base);

  float height = 0.06f + 0.14f * intensity;
  float offset_x = -height * 0.75f;
  float offset_y = height * 0.9f;

  float topLx = L + offset_x;
  float topRx = R + offset_x;
  float topBy = B + offset_y;
  float topTy = T + offset_y;

  // Top face
  glColor4f(r_base, g_base, b_base, 0.92f);
  glBegin(GL_QUADS);
  glVertex2f(topLx, topBy);
  glVertex2f(topRx, topBy);
  glVertex2f(topRx, topTy);
  glVertex2f(topLx, topTy);
  glEnd();

  // Front face
  glColor4f(r_base * 0.7f, g_base * 0.7f, b_base * 0.7f, 0.95f);
  glBegin(GL_QUADS);
  glVertex2f(topLx, topBy);
  glVertex2f(L, B);
  glVertex2f(R, B);
  glVertex2f(topRx, topBy);
  glEnd();

  // Side face
  glColor4f(r_base * 0.55f, g_base * 0.55f, b_base * 0.55f, 0.9f);
  glBegin(GL_QUADS);
  glVertex2f(topRx, topTy);
  glVertex2f(R, T);
  glVertex2f(R, B);
  glVertex2f(topRx, topBy);
  glEnd();

  // Highlight outline
  glLineWidth(1.2f);
  glColor4f(r_base * 1.1f, g_base * 1.1f, b_base * 1.1f, 0.8f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(topLx, topBy);
  glVertex2f(topRx, topBy);
  glVertex2f(topRx, topTy);
  glVertex2f(topLx, topTy);
  glEnd();
}

inline void drawCoverage(const Maze &mz, const std::vector<float> &coverage) {
  if (coverage.empty())
    return;

  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
  for (int r = 0; r < mz.H; ++r) {
    for (int c = 0; c < mz.W; ++c) {
      float level = coverage[r * mz.W + c];
      if (level <= 0.001f)
        continue;

      float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;
      float intensity = std::clamp(level, 0.0f, 1.0f);
      drawThermalPrism(L, B, R, T, intensity);
    }
  }
}

inline void drawThermalCanvas(const Maze &mz, const std::vector<float> &coverage,
                              const PathTrail &trail, double time_now,
                              float coverage_ratio) {
  glDisable(GL_DEPTH_TEST);

  glBegin(GL_QUADS);
  glColor3f(0.05f, 0.06f, 0.10f);
  glVertex2f(-1.f, -1.f);
  glColor3f(0.03f, 0.04f, 0.08f);
  glVertex2f(1.f, -1.f);
  glColor3f(0.07f, 0.05f, 0.15f);
  glVertex2f(1.f, 1.f);
  glColor3f(0.04f, 0.05f, 0.09f);
  glVertex2f(-1.f, 1.f);
  glEnd();

  drawCoverage(mz, coverage);
  drawTrailLine(mz, trail, time_now, 0.85f);
  drawGrid(mz, 0.18f);

  // Crosshair overlay
  glLineWidth(1.5f);
  glColor4f(0.2f, 0.6f, 0.9f, 0.25f);
  glBegin(GL_LINES);
  glVertex2f(-1.f, 0.f);
  glVertex2f(1.f, 0.f);
  glVertex2f(0.f, -1.f);
  glVertex2f(0.f, 1.f);
  glVertex2f(-1.f, -1.f);
  glVertex2f(1.f, 1.f);
  glVertex2f(-1.f, 1.f);
  glVertex2f(1.f, -1.f);
  glEnd();

  // Coverage bar
  float fill = std::clamp(coverage_ratio, 0.0f, 1.0f);
  float bar_height = 0.08f;
  float base_y = -1.f + bar_height * 0.35f;

  glColor4f(0.08f, 0.10f, 0.16f, 0.85f);
  glBegin(GL_QUADS);
  glVertex2f(-0.8f, base_y - bar_height * 0.5f);
  glVertex2f(0.8f, base_y - bar_height * 0.5f);
  glVertex2f(0.8f, base_y + bar_height * 0.5f);
  glVertex2f(-0.8f, base_y + bar_height * 0.5f);
  glEnd();

  float filled_half = -0.8f + 1.6f * fill;
  float rr, gg, bb;
  thermalColor(fill, rr, gg, bb);

  glColor4f(rr, gg, bb, 0.9f);
  glBegin(GL_QUADS);
  glVertex2f(-0.8f, base_y - bar_height * 0.35f);
  glVertex2f(filled_half, base_y - bar_height * 0.35f);
  glVertex2f(filled_half, base_y + bar_height * 0.35f);
  glVertex2f(-0.8f, base_y + bar_height * 0.35f);
  glEnd();

  // Marker tick
  glLineWidth(3.0f);
  glColor4f(0.95f, 0.95f, 0.95f, 0.8f);
  glBegin(GL_LINES);
  glVertex2f(filled_half, base_y - bar_height * 0.5f);
  glVertex2f(filled_half, base_y + bar_height * 0.5f);
  glEnd();

  // Outer frame
  glLineWidth(2.0f);
  glColor4f(0.12f, 0.22f, 0.35f, 0.6f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(-0.95f, -0.95f);
  glVertex2f(0.95f, -0.95f);
  glVertex2f(0.95f, 0.95f);
  glVertex2f(-0.95f, 0.95f);
  glEnd();
}

// Drawing functions
inline void drawCell(float x0, float y0, float x1, float y1, float r, float g,
                     float b) {
  glColor3f(r, g, b);
  glBegin(GL_QUADS);
  glVertex2f(x0, y0);
  glVertex2f(x1, y0);
  glVertex2f(x1, y1);
  glVertex2f(x0, y1);
  glEnd();
}

template <typename ValueProvider>
inline void drawMazeInternal(const Maze &mz, bool show_heatmap,
                             ValueProvider &&max_q_provider) {
  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;

  // Find Q-value range for heatmap normalization
  float min_q = 0.f, max_q = 0.01f;
  if (show_heatmap) {
    for (int r = 0; r < mz.H; ++r) {
      for (int c = 0; c < mz.W; ++c) {
        if (mz.at(r, c) == 0) {
          float q = max_q_provider(r, c);
          min_q = std::min(min_q, q);
          max_q = std::max(max_q, q);
        }
      }
    }
  }

  for (int r = 0; r < mz.H; ++r) {
    for (int c = 0; c < mz.W; ++c) {
      float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;

      if (mz.at(r, c) == 1) {
        // Wall with subtle gradient
        float depth = 0.1f + 0.02f * std::sin(r * 0.5f + c * 0.5f);
        drawCell(L, B, R, T, depth, depth * 1.2f, depth * 1.4f);
      } else {
        if (show_heatmap) {
          // Draw smooth Q-value heatmap
          float q = max_q_provider(r, c);
          float normalized =
              (max_q > min_q) ? (q - min_q) / (max_q - min_q) : 0.f;

          // Smooth color gradient: Blue -> Cyan -> Green -> Yellow -> Red
          float rr, gg, bb;
          if (normalized < 0.25f) {
            float t = normalized / 0.25f;
            rr = 0.0f;
            gg = 0.2f + t * 0.4f;
            bb = 0.4f + t * 0.2f;
          } else if (normalized < 0.5f) {
            float t = (normalized - 0.25f) / 0.25f;
            rr = t * 0.2f;
            gg = 0.6f + t * 0.2f;
            bb = 0.6f - t * 0.6f;
          } else if (normalized < 0.75f) {
            float t = (normalized - 0.5f) / 0.25f;
            rr = 0.2f + t * 0.6f;
            gg = 0.8f - t * 0.2f;
            bb = 0.0f;
          } else {
            float t = (normalized - 0.75f) / 0.25f;
            rr = 0.8f + t * 0.2f;
            gg = 0.6f - t * 0.6f;
            bb = 0.0f;
          }

          drawCell(L, B, R, T, rr, gg, bb);
        } else {
          // Darker floor
          drawCell(L, B, R, T, 0.02f, 0.03f, 0.05f);
        }
      }
    }
  }

  // Draw start with pulsing glow
  {
    int r = mz.start.first, c = mz.start.second;
    float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;
    float pulse = 0.7f + 0.3f * std::sin(glfwGetTime() * 2.0f);
    drawCell(L, B, R, T, 0.0f, 0.3f * pulse, 0.15f * pulse);
  }

  // Draw goal with pulsing glow
  {
    int r = mz.goal.first, c = mz.goal.second;
    float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;
    float pulse = 0.7f + 0.3f * std::sin(glfwGetTime() * 3.0f);
  	drawCell(L, B, R, T, 0.35f * pulse, 0.05f, 0.0f);
  }

}

template <typename BestActionProvider>
inline void drawPolicyArrows(const Maze &mz, bool enabled,
                             BestActionProvider &&best_action_fn) {
  if (!enabled)
    return;

  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
  glLineWidth(2.5f);

  for (int r = 0; r < mz.H; ++r) {
    for (int c = 0; c < mz.W; ++c) {
      if (mz.at(r, c) == 0 &&
          !(r == mz.start.first && c == mz.start.second) &&
          !(r == mz.goal.first && c == mz.goal.second)) {
        int best = best_action_fn(r, c);
        float cx = -1.f + c * cw + cw * 0.5f;
        float cy = 1.f - r * ch - ch * 0.5f;
        float dx = 0.f, dy = 0.f;

        if (best == 0)
          dy = ch * 0.3f; // up
        else if (best == 1)
          dy = -ch * 0.3f; // down
        else if (best == 2)
          dx = -cw * 0.3f; // left
        else if (best == 3)
          dx = cw * 0.3f; // right

        glColor4f(1.0f, 1.0f, 1.0f, 0.7f);
        glBegin(GL_LINES);
        glVertex2f(cx, cy);
        glVertex2f(cx + dx, cy + dy);
        glEnd();

        glBegin(GL_TRIANGLES);
        float size = 0.04f;
        if (best == 0) {
          glVertex2f(cx + dx, cy + dy);
          glVertex2f(cx + dx - size, cy + dy - size * 1.5f);
          glVertex2f(cx + dx + size, cy + dy - size * 1.5f);
        } else if (best == 1) {
          glVertex2f(cx + dx, cy + dy);
          glVertex2f(cx + dx - size, cy + dy + size * 1.5f);
          glVertex2f(cx + dx + size, cy + dy + size * 1.5f);
        } else if (best == 2) {
          glVertex2f(cx + dx, cy + dy);
          glVertex2f(cx + dx + size * 1.5f, cy + dy - size);
          glVertex2f(cx + dx + size * 1.5f, cy + dy + size);
        } else {
          glVertex2f(cx + dx, cy + dy);
          glVertex2f(cx + dx - size * 1.5f, cy + dy - size);
          glVertex2f(cx + dx - size * 1.5f, cy + dy + size);
        }
        glEnd();
      }
    }
  }
}

inline void drawMaze(const Maze &mz, DQNAgent *agent = nullptr,
                     bool show_heatmap = false) {
  bool heatmap_enabled = show_heatmap && agent;
  auto max_q_provider = [&](int r, int c) -> float {
    return agent->max_q(mz, r, c);
  };
  drawMazeInternal(mz, heatmap_enabled, max_q_provider);
  drawPolicyArrows(mz, heatmap_enabled,
                   [&](int r, int c) { return agent->best_action(mz, r, c); });
}

inline void drawAgent(const Maze &mz, int ar, int ac, float eps,
                      bool victory = false) {
  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;
  float L = -1.f + ac * cw, R = L + cw, T = 1.f - ar * ch, B = T - ch;
  float pad = 0.15f;
  float l = L + pad * cw, r = R - pad * cw, t = T - pad * ch, b = B + pad * ch;

  if (victory) {
    float pulse = 0.5f + 0.5f * std::sin(glfwGetTime() * 10.0f);
    glColor3f(1.0f * pulse, 0.8f * pulse, 0.0f);
  } else {
    float pulse = 0.5f + 0.5f * std::sin(glfwGetTime() * 4.0f);
    float col = 1.0f - (eps * 0.8f);
    glColor3f(0.7f * col, 0.8f * col, 1.0f * pulse);
  }

  glBegin(GL_QUADS);
  glVertex2f(l, b);
  glVertex2f(r, b);
  glVertex2f(r, t);
  glVertex2f(l, t);
  glEnd();
}
