#pragma once

#include <cstdint>
#include <vector>

#include "Libraries/Geometry/Plane.hpp"
#include "Libraries/Geometry/Point.hpp"

namespace swnumeric {

struct TriangleKernel {
  Point3D v0, v1, v2;

  inline Point3D areaVector() const { return cross(v1 - v0, v2 - v0); }
  inline double area() const { return norm(cross(v1 - v0, v2 - v0)); }

  inline void swapOrder() {
    Point3D temp = v1;
    v1 = v2;
    v2 = temp;
  }

  inline PlaneKernel getPlane() const {
    Point3D normal = areaVector();
    return PlaneKernel(normal, v0);
  }
};

struct Triangle {
  uint64_t vtx[3];

  inline Point3D areaVector(const std::vector<Point3D>& points) const {
    return cross(points[vtx[1]] - points[vtx[0]],
                 points[vtx[2]] - points[vtx[0]]);
  }
  inline double area(const std::vector<Point3D>& points) const {
    return norm(areaVector(points));
  }

  inline void swapOrder() {
    uint64_t temp = vtx[1];
    vtx[1] = vtx[2];
    vtx[2] = temp;
  }

  inline PlaneKernel getPlane(const std::vector<Point3D>& points) const {
    Point3D normal = areaVector(points);
    return PlaneKernel(normal, points[vtx[0]]);
  }
};

}  // namespace swnumeric
