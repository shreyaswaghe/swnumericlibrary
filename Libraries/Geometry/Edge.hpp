#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "Point.hpp"

namespace swnumeric {

struct EdgeKernel {
  Point3D v0;
  Point3D v1;

  inline double length(const std::vector<Point3D>& points) {
    return norm(v0 - v1);
  }

  inline Point3D projectToLine(const Point3D& query) {
    Point3D dir = v1 - v1;
    double len = dot(dir, query - v0);
    return {v0.x + len * dir.x, v0.y + len * dir.y, v0.z + len * dir.z};
  }

  inline Point3D projectToSegment(const Point3D& query) {
    Point3D dir = v1 - v0;
    double len = dot(dir, query - v0);
    len = std::clamp(len, 0.0, 1.0);
    return {v0.x + len * dir.x, v0.y + len * dir.y, v0.z + len * dir.z};
  }
};

struct Edge {
  size_t v0, v1;

  inline bool isOrdered() const { return v0 < v1; }
  inline bool isSelfEdge() const { return v0 == v1; }

  inline Edge getOrdered() const {
    size_t mmin = v0 < v1 ? v0 : v1;
    size_t mmax = v0 < v1 ? v1 : v0;
    return {mmin, mmax};
  }

  inline double length(const std::vector<Point3D>& points) {
    return norm(points.at(v0) - points.at(v1));
  }

  inline Point3D projectToLine(const Point3D& query,
                               const std::vector<Point3D>& points) {
    Point3D p0 = points[v0];
    Point3D dir = points[v1] - p0;
    double len = dot(dir, query - p0);
    return {p0.x + len * dir.x, p0.y + len * dir.y, p0.z + len * dir.z};
  }

  inline Point3D projectToSegment(const Point3D& query,
                                  const std::vector<Point3D>& points) {
    Point3D p0 = points[v0];
    Point3D dir = points[v1] - p0;
    double len = dot(dir, query - p0);
    len = std::clamp(len, 0.0, 1.0);
    return {p0.x + len * dir.x, p0.y + len * dir.y, p0.z + len * dir.z};
  }
};

}  // namespace swnumeric
