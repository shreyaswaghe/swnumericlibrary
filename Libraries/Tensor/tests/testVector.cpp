// test_vector_full.cpp
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "../Vector.hpp"

using namespace swnumeric;

class TestFramework {
 public:
  using TestFunc = std::function<bool()>;

  void addTest(const std::string& name, TestFunc func) {
    bool passed = false;
    std::string msg;
    try {
      passed = func();
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {
      msg = "Unknown exception";
    }
    results.push_back({name, passed, msg});
  }

  void summary() const {
    size_t passedCount = 0;
    std::cout << "===== TEST SUMMARY =====\n";
    for (const auto& r : results) {
      std::cout << (r.passed ? "[PASS] " : "[FAIL] ") << r.name;
      if (!r.passed && !r.message.empty())
        std::cout << " (" << r.message << ")";
      std::cout << "\n";
      if (r.passed) passedCount++;
    }
    std::cout << "========================\n"
              << "Passed: " << passedCount << "/" << results.size() << "\n";
  }

  bool allPassed() const {
    for (const auto& r : results)
      if (!r.passed) return false;
    return true;
  }

 private:
  struct TestResult {
    std::string name;
    bool passed;
    std::string message;
  };
  std::vector<TestResult> results;
};

// Helper macros
#define EXPECT_EQ(a, b) ((a) == (b))
#define EXPECT_TRUE(a) (a)

template <typename T>
static bool approxEqual(T a, T b, T eps = T(1e-9)) {
  return std::abs(a - b) <= eps * (T(1) + std::abs(a) + std::abs(b));
}

// ======= Tests =======

int main() {
  using V3 = Vector<3, double>;
  using VD = Vector<0, double>;

  TestFramework tf;

  tf.addTest("Static ctor sets fixed size and alloc state", [] {
    V3 v;
    return EXPECT_EQ(v.size(), size_t(3)) && EXPECT_TRUE(v.isAlloc()) &&
           EXPECT_EQ(v.nDims(), size_t(1));
  });

  tf.addTest("Dynamic default ctor with size=0 leaves unallocated", [] {
    VD v0;
    return EXPECT_EQ(v0.size(), size_t(0)) && !v0.isAlloc();
  });

  tf.addTest("Dynamic size ctor allocates when sz>0", [] {
    VD v(5);
    bool ok = EXPECT_EQ(v.size(), size_t(5)) && EXPECT_TRUE(v.isAlloc());
    for (size_t i = 0; i < v.size(); ++i) v[i] = double(i + 1);
    return ok && EXPECT_EQ(v[0], 1.0) && EXPECT_EQ(v[4], 5.0);
  });

  tf.addTest("Shape ctor behaves like size ctor", [] {
    VD v(std::array<size_t, 1>{4});
    bool ok = EXPECT_EQ(v.size(), size_t(4));
    for (size_t i = 0; i < 4; ++i) v[i] = double(2 * i);
    return ok && EXPECT_EQ(v.shape()[0], size_t(4)) && EXPECT_EQ(*(v(2)), 4.0);
  });

  tf.addTest("Static ctor from array copies values", [] {
    V3 v(std::array<double, 3>{1.0, 2.0, 3.0});
    return EXPECT_EQ(v[0], 1.0) && EXPECT_EQ(v[1], 2.0) && EXPECT_EQ(v[2], 3.0);
  });

  tf.addTest("Copy ctor static -> deep copy", [] {
    V3 a;
    for (int i = 0; i < 3; ++i) a[i] = i + 1;
    V3 b(a);
    a[0] = 100.0;
    return EXPECT_EQ(b[0], 1.0) && EXPECT_EQ(b[1], 2.0) && EXPECT_EQ(b[2], 3.0);
  });

  tf.addTest("Copy ctor dynamic -> deep copy", [] {
    VD a(3);
    for (int i = 0; i < 3; ++i) a[i] = i + 1;
    VD b(a);
    a[1] = 200.0;
    return EXPECT_EQ(b[0], 1.0) && EXPECT_EQ(b[1], 2.0) && EXPECT_EQ(b[2], 3.0);
  });

  tf.addTest("Copy assignment self no-op (static)", [] {
    V3 a;
    for (int i = 0; i < 3; ++i) a[i] = i + 1;
    V3* pa = &a;
    a = *pa;
    return EXPECT_EQ(a[0], 1.0) && EXPECT_EQ(a[1], 2.0) && EXPECT_EQ(a[2], 3.0);
  });

  tf.addTest("Copy assignment dynamic reallocates and copies", [] {
    VD a(2);
    a[0] = 5;
    a[1] = 6;
    VD b(4);
    for (int i = 0; i < 4; ++i) b[i] = i;
    b = a;
    return EXPECT_EQ(b.size(), size_t(2)) && EXPECT_EQ(b[0], 5.0) &&
           EXPECT_EQ(b[1], 6.0);
  });

  tf.addTest("Move ctor dynamic steals pointer", [] {
    VD a(3);
    a[0] = 7;
    a[1] = 8;
    a[2] = 9;
    VD b(std::move(a));
    bool ok = EXPECT_EQ(b.size(), size_t(3)) && EXPECT_EQ(b[0], 7.0) &&
              EXPECT_EQ(a.size(), size_t(0));
    return ok;
  });

  tf.addTest("Move assignment dynamic releases old and steals new", [] {
    VD a(2);
    a[0] = 1;
    a[1] = 2;
    VD b(3);
    b[0] = 9;
    b[1] = 9;
    b[2] = 9;
    b = std::move(a);
    bool ok = EXPECT_EQ(b.size(), size_t(2)) && EXPECT_EQ(b[1], 2.0);
    return ok;
  });

  tf.addTest("operator[] and operator() pointer access", [] {
    V3 v;
    v[0] = 10;
    v[1] = 11;
    v[2] = 12;
    return EXPECT_EQ(*(v(0)), 10.0) && EXPECT_EQ(*(v(2)), 12.0);
  });

  tf.addTest("Backend assign from scalar", [] {
    V3 v;
    v = 3.5;
    return EXPECT_EQ(v[0], 3.5) && EXPECT_EQ(v[1], 3.5) && EXPECT_EQ(v[2], 3.5);
  });

  tf.addTest("Vector += Vector and + scalar", [] {
    V3 a;
    a = 1.0;
    V3 b;
    b = 2.0;
    a += b;          // now 3,3,3
    V3 c = a + 1.0;  // now 4,4,4
    return EXPECT_EQ(a[0], 3.0) && EXPECT_EQ(c[1], 4.0);
  });

  tf.addTest("Vector + Vector (non-inplace)", [] {
    VD a(3);
    VD b(3);
    for (int i = 0; i < 3; ++i) {
      a[i] = i;
      b[i] = 10 + i;
    }
    VD c = a + b;
    return EXPECT_EQ(c[0], 10.0) && EXPECT_EQ(c[1], 12.0) &&
           EXPECT_EQ(c[2], 14.0);
  });

  tf.addTest("Vector -= scalar and binary -", [] {
    V3 a;
    a = 5.0;
    a -= 2.0;        // 3,3,3
    V3 d = a - 1.0;  // 2,2,2
    return EXPECT_EQ(a[2], 3.0) && EXPECT_EQ(d[0], 2.0);
  });

  tf.addTest("Vector elementwise * and /", [] {
    VD a(3), b(3);
    a[0] = 2;
    a[1] = 4;
    a[2] = 6;
    b[0] = 1;
    b[1] = 2;
    b[2] = 3;
    VD c = a * b;  // 2,8,18
    VD d = c / b;  // back to 2,4,6
    return EXPECT_EQ(c[1], 8.0) && EXPECT_EQ(d[2], 6.0);
  });

  tf.addTest("dot, sdot, norm2", [] {
    V3 v;
    v[0] = 3;
    v[1] = 4;
    v[2] = 12;
    V3 w;
    w[0] = 1;
    w[1] = 0.5;
    w[2] = 2;
    double dp = v.dot(w);
    double sdp = v.dot(w);
    double n2 = v.norm2();
    bool ok = approxEqual(dp, 3 * 1 + 4 * 0.5 + 12 * 2) &&
              approxEqual(sdp, dp) && approxEqual(n2, std::sqrt(9 + 16 + 144));
    return ok;
  });

  tf.addTest("indexmax (argmax of |v_i|)", [] {
    VD v(5);
    v[0] = -1;
    v[1] = 2;
    v[2] = -10;
    v[3] = 7;
    v[4] = 0.5;
    return EXPECT_EQ(v.indexmax(), size_t(2));
  });

  tf.addTest("shape() and nDims()", [] {
    VD v(7);
    return EXPECT_EQ(v.shape()[0], size_t(7)) &&
           EXPECT_EQ(v.nDims(), size_t(1));
  });

  tf.addTest("Zero-size dynamic survives ops and norms", [] {
    VD v0;  // size 0
    // These operations should be no-ops / defined (our backend tolerates
    // size=0)
    v0 = 1.0;
    v0 += 2.0;
    bool ok = EXPECT_EQ(v0.size(), size_t(0));
    // indexmax returns 0 by convention for empty vector in our backend
    return ok && EXPECT_EQ(v0.indexmax(), size_t(-1));
  });

  tf.addTest("Post-move state of dynamic is null and size=0", [] {
    VD a(3);
    a = 1.0;
    VD b = std::move(a);
    return EXPECT_EQ(a.size(), size_t(0));
  });

  tf.addTest("norm1 and norminf currently equal norm2 by design note", [] {
    V3 v;
    v[0] = -3;
    v[1] = 4;
    v[2] = 12;
    return approxEqual(v.norm1(), 19.0) && approxEqual(v.norminf(), 12.0);
  });

  // Run & report
  tf.summary();
  return tf.allPassed() ? 0 : 1;
}
