#pragma once

#include "Libraries/Geometry/Point.hpp"
namespace swnumeric {

struct PlaneKernel {
  // represent plane as
  // n^T (x - x_0)
  Point3D normal;
  double distance;

  PlaneKernel(const Point3D& normal, const Point3D& pointOnPlane) {
    this->normal = normal;
    normalizeInPlace(this->normal);
    this->distance = dot(this->normal, pointOnPlane);
  }

  PlaneKernel(const Point3D& normal, const double& distance) {
    this->normal = normal;
    normalizeInPlace(this->normal);
    double mag = norm(normal);
    this->distance = distance / mag;
  }

  inline Point3D getProjection(const Point3D& query) {
    double normalCoord = dot(query, normal);
    Point3D proj = query;
    proj -= normalCoord * query;
    return proj;
  }
};

}  // namespace swnumeric
