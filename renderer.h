#pragma once
#include "env.h"
#include "qagent.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <deque>

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
    float cw = 2.0f / mz.W, ch = 2.0f / mz.H;

    for (const auto &p : points) {
      double age = current_time - p.time;
      if (age > fade_time)
        continue;

      float alpha = 1.0f - (age / fade_time);
      float L = -1.f + p.c * cw, R = L + cw, T = 1.f - p.r * ch, B = T - ch;
      float pad = 0.3f;
      float l = L + pad * cw, r = R - pad * cw, t = T - pad * ch,
            b = B + pad * ch;

      glColor4f(0.2f, 0.6f, 1.0f, alpha * 0.5f);
      glBegin(GL_QUADS);
      glVertex2f(l, b);
      glVertex2f(r, b);
      glVertex2f(r, t);
      glVertex2f(l, t);
      glEnd();
    }
  }
};

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

inline void drawMaze(const Maze &mz, const QAgent *agent = nullptr,
                     bool show_heatmap = false) {
  float cw = 2.0f / mz.W, ch = 2.0f / mz.H;

  // Find Q-value range for heatmap normalization
  float min_q = 0.f, max_q = 0.f;
  if (show_heatmap && agent) {
    for (int r = 0; r < mz.H; ++r) {
      for (int c = 0; c < mz.W; ++c) {
        if (mz.at(r, c) == 0) {
          float q = agent->max_q(r, c);
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
        drawCell(L, B, R, T, 0.1f, 0.12f, 0.14f);
      } else {
        if (show_heatmap && agent) {
          // Draw Q-value heatmap
          float q = agent->max_q(r, c);
          float normalized =
              (max_q > min_q) ? (q - min_q) / (max_q - min_q) : 0.f;
          float hue = 0.6f - normalized * 0.6f; // Blue (cold) to Red (hot)
          float rr, gg, bb;
          hsv_to_rgb(hue, 0.8f, 0.4f + normalized * 0.4f, rr, gg, bb);
          drawCell(L, B, R, T, rr, gg, bb);
        } else {
          drawCell(L, B, R, T, 0.03f, 0.04f, 0.06f);
        }
      }
    }
  }

  // Draw start
  {
    int r = mz.start.first, c = mz.start.second;
    float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;
    drawCell(L, B, R, T, 0.0f, 0.25f, 0.10f);
  }

  // Draw goal
  {
    int r = mz.goal.first, c = mz.goal.second;
    float L = -1.f + c * cw, R = L + cw, T = 1.f - r * ch, B = T - ch;
    drawCell(L, B, R, T, 0.25f, 0.05f, 0.0f);
  }

  // Draw policy arrows if heatmap is shown
  if (show_heatmap && agent) {
    glLineWidth(2.0f);
    for (int r = 0; r < mz.H; ++r) {
      for (int c = 0; c < mz.W; ++c) {
        if (mz.at(r, c) == 0 &&
            !(r == mz.start.first && c == mz.start.second) &&
            !(r == mz.goal.first && c == mz.goal.second)) {
          int best = agent->best_action(r, c);
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

          glColor3f(1.0f, 1.0f, 1.0f);
          glBegin(GL_LINES);
          glVertex2f(cx, cy);
          glVertex2f(cx + dx, cy + dy);
          glEnd();

          // Arrow head
          glBegin(GL_TRIANGLES);
          float size = 0.05f;
          if (best == 0) {
            glVertex2f(cx + dx, cy + dy);
            glVertex2f(cx + dx - size, cy + dy - size);
            glVertex2f(cx + dx + size, cy + dy - size);
          } else if (best == 1) {
            glVertex2f(cx + dx, cy + dy);
            glVertex2f(cx + dx - size, cy + dy + size);
            glVertex2f(cx + dx + size, cy + dy + size);
          } else if (best == 2) {
            glVertex2f(cx + dx, cy + dy);
            glVertex2f(cx + dx + size, cy + dy - size);
            glVertex2f(cx + dx + size, cy + dy + size);
          } else {
            glVertex2f(cx + dx, cy + dy);
            glVertex2f(cx + dx - size, cy + dy - size);
            glVertex2f(cx + dx - size, cy + dy + size);
          }
          glEnd();
        }
      }
    }
  }
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
