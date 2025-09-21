#pragma once

#include <cmath>

namespace swnumeric {

struct Point3D {
  double x, y, z;

  Point3D& operator+=(const Point3D& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Point3D& operator-=(const Point3D& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  Point3D& operator*=(const Point3D& other) {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
  }

  Point3D& operator/=(const Point3D& other) {
    x /= other.x;
    y /= other.y;
    z /= other.z;
    return *this;
  }

  Point3D& operator+=(const double& other) {
    x += other;
    y += other;
    z += other;
    return *this;
  }

  Point3D& operator-=(const double& other) {
    x -= other;
    y -= other;
    z -= other;
    return *this;
  }

  Point3D& operator*=(const double& other) {
    x *= other;
    y *= other;
    z *= other;
    return *this;
  }

  Point3D& operator/=(const double& other) {
    x /= other;
    y /= other;
    z /= other;
    return *this;
  }
};

inline Point3D operator+(const Point3D& a, const Point3D& b) {
  Point3D c = a;
  c += b;
  return c;
}

inline Point3D operator+(const Point3D& a, const double& b) {
  Point3D c = a;
  c += b;
  return c;
}

inline Point3D operator+(const double& a, const Point3D& b) {
  Point3D c = b;
  c += a;
  return c;
}

inline Point3D operator-(const Point3D& a, const Point3D& b) {
  Point3D c = a;
  c -= b;
  return c;
}

inline Point3D operator-(const Point3D& a, const double& b) {
  Point3D c = a;
  c -= b;
  return c;
}

inline Point3D operator-(const double& a, const Point3D& b) {
  Point3D c = b;
  c -= a;
  return c;
}

inline Point3D operator*(const Point3D& a, const Point3D& b) {
  Point3D c = a;
  c *= b;
  return c;
}

inline Point3D operator*(const double& a, const Point3D& b) {
  Point3D c = b;
  c *= a;
  return c;
}

inline Point3D operator*(const Point3D& a, const double& b) {
  Point3D c = a;
  c *= b;
  return c;
}

inline Point3D operator/(const Point3D& a, const Point3D& b) {
  Point3D c = a;
  c /= b;
  return c;
}

inline Point3D operator/(const double& a, const Point3D& b) {
  Point3D c = b;
  c /= a;
  return c;
}

inline Point3D operator/(const Point3D& a, const double& b) {
  Point3D c = a;
  c /= b;
  return c;
}

inline double normSq(const Point3D& x) {
  double xc = fabs(x.x), yc = fabs(x.y), zc = fabs(x.z);
  double mmax = fmax(fmax(x.x, x.y), x.z);
  xc /= mmax;
  yc /= mmax;
  zc /= mmax;

  return (mmax * mmax) * (xc * xc + yc * yc + zc * zc);
}

inline double norm(const Point3D& x) {
  double xc = fabs(x.x), yc = fabs(x.y), zc = fabs(x.z);
  double mmax = fmax(fmax(x.x, x.y), x.z);
  xc /= mmax;
  yc /= mmax;
  zc /= mmax;

  return mmax * sqrt((xc * xc + yc * yc + zc * zc));
}

inline Point3D normalize(const Point3D& p) {
  double mag = norm(p);

  Point3D vec = p;
  if (mag != 0.0) {
    vec /= mag;
  }
  return vec;
}

inline void normalizeInPlace(Point3D& p) {
  double mag = norm(p);

  if (mag != 0.0) {
    p /= mag;
  }
}

inline double dot(const Point3D& x, const Point3D& y) {
  return x.x * y.x + x.y * y.y + x.z * y.z;
}

inline double cosAngle(const Point3D& x, const Point3D& y) {
  return dot(x, y) / norm(x) / norm(y);
}

inline Point3D cross(const Point3D& p1, const Point3D& p2) {
  double x = p1.y * p2.z - p1.z * p2.y;
  double y = p1.x * p2.z - p1.z * p2.x;
  double z = p1.x * p2.y - p1.y * p2.x;
  return {.x = x, .y = -y, .z = z};
}

}  // namespace swnumeric
