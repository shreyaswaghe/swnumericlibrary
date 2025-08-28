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

int main() {
  TestFramework tf;
  std::mt19937 rng(42);  // deterministic RNG
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  auto testMatrixOps = [&tf](auto name, auto testFunc) {
    tf.addTest(name, testFunc);
  };

  // --- 1. Static-size matrix assignment ---
  testMatrixOps("Static Double Assignment", []() {
    Matrix<2, 2, double, CblasBackend> A;
    A = 1.0;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 1.0)) return false;
    return true;
  });

  // --- 2. Static-size arithmetic ---
  testMatrixOps("Static Double Arithmetic", []() {
    Matrix<2, 2, double, CblasBackend> A, B;
    A = 2.0;
    B = 3.0;

    auto C = A + B;
    auto D = A - B;
    auto E = A * B;
    auto F = A / B;
    for (size_t i = 0; i < A.size(); i++) {
      if (!EXPECT_EQ(C[i], 5.0)) return false;
      if (!EXPECT_EQ(D[i], -1.0)) return false;
      if (!EXPECT_EQ(E[i], 6.0)) return false;
      if (!EXPECT_TRUE(std::fabs(F[i] - 2.0 / 3.0) < 1e-12)) return false;
    }
    return true;
  });

  // --- 3. Dynamic-size matrix assignment ---
  testMatrixOps("Dynamic Double Assignment", []() {
    Matrix<0, 0, double, CblasBackend> A(3, 4);
    A = 2.5;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 2.5)) return false;
    return true;
  });

  // --- 4. Dynamic arithmetic ---
  testMatrixOps("Dynamic Double Arithmetic", []() {
    Matrix<0, 0, double, CblasBackend> A(2, 3), B(2, 3);
    A = 4.0;
    B = 1.0;
    auto C = A + B;
    auto D = A - B;
    for (size_t i = 0; i < A.size(); i++) {
      if (!EXPECT_EQ(C[i], 5.0)) return false;
      if (!EXPECT_EQ(D[i], 3.0)) return false;
    }
    return true;
  });

  // --- 5. Dot product ---
  testMatrixOps("Dot Product", []() {
    Matrix<2, 2, double, CblasBackend> A, B;
    A = 1.0;
    B = 2.0;
    double result = A.dot(B);
    // dot( [1,1,1,1], [2,2,2,2] ) = 4 * 2 = 8
    return EXPECT_EQ(result, 8.0);
  });

  // --- 6. Frobenius norm ---
  testMatrixOps("Frobenius Norm", []() {
    Matrix<2, 2, double, CblasBackend> A;
    A = 3.0;
    double n = A.normfro();
    // sqrt(3^2 + 3^2 + 3^2 + 3^2) = sqrt(36) = 6
    return EXPECT_TRUE(std::fabs(n - 6.0) < 1e-12);
  });

  // --- 7. Random large dynamic matrix ---
  testMatrixOps("Large Dynamic Matrix", [&]() {
    const size_t R = 20, C = 30;
    Matrix<0, 0, double, CblasBackend> A(R, C), B(R, C);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = dist(rng);
      B[i] = dist(rng);
    }
    auto Cmat = A + B;
    for (size_t i = 0; i < A.size(); i++) {
      if (!EXPECT_TRUE(std::fabs(Cmat[i] - (A[i] + B[i])) < 1e-12))
        return false;
    }
    return true;
  });

  // --- 8. Scalar ops ---
  testMatrixOps("Scalar Ops", []() {
    Matrix<2, 2, double, CblasBackend> A;
    A = 1.0;
    A *= 2.0;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 2.0)) return false;
    A += 3.0;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 5.0)) return false;
    A -= 1.0;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 4.0)) return false;
    A /= 2.0;
    for (size_t i = 0; i < A.size(); i++)
      if (!EXPECT_EQ(A[i], 2.0)) return false;
    return true;
  });

  // --- 9. Indexing and shape ---
  testMatrixOps("Shape and Indexing", []() {
    Matrix<3, 4, double, CblasBackend> A;
    if (!EXPECT_EQ(A.shape()[0], 3)) return false;
    if (!EXPECT_EQ(A.shape()[1], 4)) return false;
    if (!EXPECT_EQ(A.nDims(), 2)) return false;
    if (!EXPECT_EQ(A.lda(), 3)) return false;
    return true;
  });

  tf.summary();
  return tf.allPassed() ? 0 : 1;
}
