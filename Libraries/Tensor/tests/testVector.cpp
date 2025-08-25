// test_vector_full.cpp
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "../Vector.hpp"

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

int main() {
  TestFramework tf;
  std::mt19937 rng(42);  // deterministic RNG
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  auto testVectorOps = [&tf](auto name, auto testFunc) {
    tf.addTest(name, testFunc);
  };

  // --- 1. Static-size double vector ---
  testVectorOps("Static Double Assignment", []() {
    Vector<3, double, CblasBackend> a;
    a = 1.0;
    for (size_t i = 0; i < a.size(); i++)
      if (!EXPECT_EQ(a[i], 1.0)) return false;
    return true;
  });

  testVectorOps("Static Double Arithmetic", []() {
    Vector<3, double, CblasBackend> a, b;
    a = 2.0;
    b = 3.0;

    auto c = a + b;
    auto d = a - b;
    auto e = a * b;
    auto f = a / b;
    for (size_t i = 0; i < 3; i++) {
      if (!EXPECT_EQ(c[i], 5.0)) return false;
      if (!EXPECT_EQ(d[i], -1.0)) return false;
      if (!EXPECT_EQ(e[i], 6.0)) return false;
      if (!EXPECT_TRUE(std::fabs(f[i] - 2.0 / 3.0) < 1e-12)) return false;
    }
    a += b;
    a -= b;
    a *= 2.0;
    a /= 2.0;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(a[i], 2.0)) return false;
    return true;
  });

  // --- 2. Dynamic-size double vector ---
  testVectorOps("Dynamic Double Assignment", []() {
    Vector<0, double, CblasBackend> a(5);
    a = 2.0;
    for (size_t i = 0; i < a.size(); i++)
      if (!EXPECT_EQ(a[i], 2.0)) return false;
    return true;
  });

  testVectorOps("Dynamic Double Arithmetic", [&]() {
    Vector<0, double, CblasBackend> a(4), b(4);
    a = 3.0;
    b = 2.0;
    auto c = a + b;
    auto d = a - b;
    auto e = a * b;
    auto f = a / b;
    for (size_t i = 0; i < 4; i++) {
      if (!EXPECT_EQ(c[i], 5.0)) return false;
      if (!EXPECT_EQ(d[i], 1.0)) return false;
      if (!EXPECT_EQ(e[i], 6.0)) return false;
      if (!EXPECT_TRUE(std::fabs(f[i] - 1.5) < 1e-12)) return false;
    }
    return true;
  });

  // --- 3. Float vectors ---
  testVectorOps("Float Vector Ops", []() {
    Vector<3, float, CblasBackend> a, b;
    a = 2.0f;
    b = 3.0f;
    auto c = a + b;
    auto d = a - b;
    auto e = a * b;
    auto f = a / b;
    for (size_t i = 0; i < 3; i++) {
      if (!EXPECT_TRUE(std::fabs(c[i] - 5.0f) < 1e-6f)) return false;
      if (!EXPECT_TRUE(std::fabs(d[i] + 1.0f) < 1e-6f)) return false;
      if (!EXPECT_TRUE(std::fabs(e[i] - 6.0f) < 1e-6f)) return false;
      if (!EXPECT_TRUE(std::fabs(f[i] - 2.0f / 3.0f) < 1e-6f)) return false;
    }
    return true;
  });

  // --- 4. SCALAR_TIMES_VECTOR expressions ---
  testVectorOps("Scalar-Times-Vector", []() {
    Vector<3, double, CblasBackend> a, b;
    a = 2.0;
    b = 3.0;
    auto expr = 2.0 * b;
    a += expr;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(a[i], 2.0 + 2.0 * 3.0)) return false;
    a *= expr;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(a[i], (2.0 + 6.0) * 6.0)) return false;
    return true;
  });

  // --- 5. Large dynamic vector ---
  testVectorOps("Large Dynamic Vector", [&]() {
    const size_t N = 1000;
    Vector<0, double, CblasBackend> a(N), b(N);
    a = 1.0;
    b = 2.0;
    a += b;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_EQ(a[i], 3.0)) return false;
    a *= 2.0;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_EQ(a[i], 6.0)) return false;
    return true;
  });

  // --- 6. Random dynamic vectors ---
  testVectorOps("Random Dynamic Vectors", [&]() {
    const size_t N = 1000;
    Vector<0, double, CblasBackend> a(N), b(N);
    for (size_t i = 0; i < N; i++) {
      a[i] = dist(rng);
      b[i] = dist(rng);
    }
    Vector<0, double, CblasBackend> c = a + b;
    Vector<0, double, CblasBackend> d = a - b;
    Vector<0, double, CblasBackend> e = a * b;
    for (size_t i = 0; i < N; i++) {
      if (!EXPECT_TRUE(std::fabs(c[i] - (a[i] + b[i])) < 1e-12)) return false;
      if (!EXPECT_TRUE(std::fabs(d[i] - (a[i] - b[i])) < 1e-12)) return false;
      if (!EXPECT_TRUE(std::fabs(e[i] - (a[i] * b[i])) < 1e-12)) return false;
    }
    return true;
  });

  // --- 7. Random scalar operations ---
  testVectorOps("Random Scalar Ops", [&]() {
    const size_t N = 500;
    Vector<0, double, CblasBackend> a(N);
    for (size_t i = 0; i < N; i++) a[i] = dist(rng);
    double scalar = dist(rng);

    Vector<0, double, CblasBackend> b = a;
    b *= scalar;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_TRUE(std::fabs(b[i] - a[i] * scalar) < 1e-12)) return false;

    b = a;
    b /= scalar;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_TRUE(std::fabs(b[i] - a[i] / scalar) < 1e-12)) return false;

    b = a;
    b += scalar;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_TRUE(std::fabs(b[i] - (a[i] + scalar)) < 1e-12)) return false;

    b = a;
    b -= scalar;
    for (size_t i = 0; i < N; i++)
      if (!EXPECT_TRUE(std::fabs(b[i] - (a[i] - scalar)) < 1e-12)) return false;

    return true;
  });

  testVectorOps("Scalar Zero Operations", []() {
    Vector<3, double, CblasBackend> a{{1.0, -2.0, 3.5}};
    a *= 0.0;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(a[i], 0.0)) return false;
    a = Vector<3>{{1.0, -2.0, 3.5}};
    a /= 1.0;
    if (!EXPECT_TRUE(a[0] == 1.0 && a[1] == -2.0 && a[2] == 3.5)) return false;
    return true;
  });

  testVectorOps("Negative Numbers", []() {
    Vector<3, double, CblasBackend> a{{-1.0, -2.0, -3.0}}, b{{1.0, 2.0, 3.0}};
    Vector<3, double, CblasBackend> c = a + b;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(c[i], 0.0)) return false;
    c = a - b;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(c[i], -2.0 * (i + 1))) return false;
    c = a * b;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(c[i], -1.0 * (i + 1) * (i + 1))) return false;
    c = b / a;
    for (size_t i = 0; i < 3; i++)
      if (!EXPECT_EQ(c[i], -1.0)) return false;
    return true;
  });

  tf.summary();
  return tf.allPassed() ? 0 : 1;
}
