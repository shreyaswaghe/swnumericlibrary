// test_matrix_full.cpp
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "../Matrix.hpp"  // adjust path if needed

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

int main() {
  TestFramework tf;

  // ----- Construction and Element Access -----
  tf.addTest("Default construction", [] {
    Matrix<2, 2, double> m;
    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        if (!approxEqual(m(i, j), 0.0)) return false;
    return true;
  });

  tf.addTest("Initializer array construction", [] {
    Matrix<2, 2, int> m{{{1, 2}, {3, 4}}};
    return m(0, 0) == 1 && m(0, 1) == 2 && m(1, 0) == 3 && m(1, 1) == 4;
  });

  tf.addTest("Copy constructor", [] {
    Matrix<2, 2, int> m{{1, 2}, {3, 4}};
    Matrix<2, 2, int> c(m);
    return EXPECT_EQ(c(1, 1), 4);
  });

  tf.addTest("Assignment operator", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    Matrix<2, 2, int> b;
    b = a;
    return EXPECT_EQ(b(0, 1), 2);
  });

  // ----- Elementwise Addition/Subtraction -----
  tf.addTest("Elementwise addition", [] {
    Matrix<2, 2, int> a({{1, 2}, {3, 4}});
    Matrix<2, 2, int> b = {{5, 6}, {7, 8}};
    auto c = a + b;
    return EXPECT_EQ(c(0, 0), 6) && EXPECT_EQ(c(1, 1), 12);
  });

  tf.addTest("Elementwise subtraction", [] {
    Matrix<2, 2, int> a{{5, 6}, {7, 8}};
    Matrix<2, 2, int> b{{1, 2}, {3, 4}};
    auto c = a - b;
    return EXPECT_EQ(c(0, 0), 4) && EXPECT_EQ(c(1, 1), 4);
  });

  tf.addTest("Compound elementwise +=/-=", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    Matrix<2, 2, int> b{{5, 6}, {7, 8}};
    a += b;
    if (!(a(0, 0) == 6 && a(1, 1) == 12)) return false;
    a -= b;
    return EXPECT_EQ(a(0, 0), 1) && EXPECT_EQ(a(1, 1), 4);
  });

  // ----- Elementwise Multiplication/Division -----
  tf.addTest("Elementwise multiplication", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    Matrix<2, 2, int> b{{2, 3}, {4, 5}};
    auto c = a * b;
    return EXPECT_EQ(c(0, 0), 2) && EXPECT_EQ(c(1, 1), 20);
  });

  tf.addTest("Elementwise division", [] {
    Matrix<2, 2, double> a{{2.0, 4.0}, {6.0, 8.0}};
    Matrix<2, 2, double> b{{2.0, 2.0}, {3.0, 4.0}};
    auto c = a / b;
    return approxEqual(c(0, 0), 1.0) && approxEqual(c(1, 1), 2.0);
  });

  tf.addTest("Compound elementwise *=/=", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    Matrix<2, 2, int> b{{2, 2}, {2, 2}};
    a *= b;
    if (!(a(0, 0) == 2 && a(1, 1) == 8)) return false;
    a /= b;
    return a(0, 0) == 1 && a(1, 1) == 4;
  });

  // ----- Scalar operations -----
  tf.addTest("Scalar addition/subtraction", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    a += 2;
    if (!(a(0, 0) == 3 && a(1, 1) == 6)) return false;
    a -= 1;
    return a(0, 0) == 2 && a(1, 1) == 5;
  });

  tf.addTest("Scalar multiplication/division", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    a *= 3;
    assert(a(1, 0) == 9);
    if (!(a(1, 0) == 9)) return false;
    a /= 3;
    assert(a(1, 0) == 3);
    return a(1, 0) == 3 && a(1, 1) == 4;
  });

  // ----- Norms -----
  tf.addTest("Frobenius and L2 norm", [] {
    Matrix<2, 2, double> a{{3, 4}, {0, 0}};
    return approxEqual(a.normfro(), 5.0) && approxEqual(a.norm2(), 5.0);
  });

  tf.addTest("L1 norm", [] {
    Matrix<2, 2, int> a{{1, -2}, {3, -4}};
    std::cout << a.norm1() << std::endl;
    return EXPECT_EQ(a.norm1(), 10);
  });

  tf.addTest("Infinity norm", [] {
    Matrix<2, 2, int> a{{1, -2}, {3, -4}};
    return EXPECT_EQ(a.norminf(), 4);
  });

  // ----- Index max -----
  tf.addTest("Index max", [] {
    Matrix<2, 2, int> a{{1, 5}, {3, 2}};
    return a.indexmax() == 2;  // flattened index
  });

  // ----- Edge Cases -----
  tf.addTest("Self assignment", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    a = a;
    return EXPECT_EQ(a(0, 0), 1) && EXPECT_EQ(a(1, 1), 4);
  });

  tf.addTest("Zero matrix behavior", [] {
    Matrix<2, 2, int> a{{1, 2}, {3, 4}};
    Matrix<2, 2, int> z{};
    auto c = a + z;
    return EXPECT_TRUE(c(0, 0) == a(0, 0));
  });

  tf.summary();
  return tf.allPassed() ? 0 : 1;
}
