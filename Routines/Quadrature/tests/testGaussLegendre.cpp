#include <cassert>
#include <cmath>
#include <iostream>

#include "../GaussLegendre.hpp"

using namespace swnumeric;

// Test integrand classes

struct ConstantIntegrandFactory {
  static auto get(double c) {
    return [c](double x) { return c; };
  }
};

// Helper function to check if two floating point numbers are approximately
// equal
template <typename T>
bool approx_equal(T a, T b, T tolerance = static_cast<T>(1e-12)) {
  if (std::is_same_v<T, float>) {
    tolerance = static_cast<T>(1e-6);
  }
  return std::abs(a - b) < tolerance;
}

// Test functions
void test_constant_integration() {
  std::cout << "Testing constant integration...\n";

  GaussLegendre gl;
  const auto& constant_5 = ConstantIntegrandFactory::get(5.0);

  // Integral of 5 from -1 to 1 should be 10
  double result2 = gl.eval<2>(constant_5);
  double result4 = gl.eval<4>(constant_5);
  double result8 = gl.eval<8>(constant_5);
  double result16 = gl.eval<16>(constant_5);
  double result32 = gl.eval<32>(constant_5);
  double result64 = gl.eval<64>(constant_5);

  double expected = 10.0;
  assert(approx_equal(result2, expected));
  assert(approx_equal(result4, expected));
  assert(approx_equal(result8, expected));
  assert(approx_equal(result16, expected));
  assert(approx_equal(result32, expected));
  assert(approx_equal(result64, expected));

  std::cout << "✓ Constant integration tests passed\n";
}

void test_linear_integration() {
  std::cout << "Testing linear integration...\n";

  GaussLegendre gl;
  const auto linear = [](const double x) { return x; };

  // Integral of x from -1 to 1 should be 0 (odd function)
  double result2 = gl.eval<2>(linear);
  double result4 = gl.eval<4>(linear);
  double result8 = gl.eval<8>(linear);
  double result16 = gl.eval<16>(linear);
  double result32 = gl.eval<32>(linear);
  double result64 = gl.eval<64>(linear);

  double expected = 0.0;
  assert(approx_equal(result2, expected));
  assert(approx_equal(result4, expected));
  assert(approx_equal(result8, expected));
  assert(approx_equal(result16, expected));
  assert(approx_equal(result32, expected));
  assert(approx_equal(result64, expected));

  std::cout << "✓ Linear integration tests passed\n";
}

void test_quadratic_integration() {
  std::cout << "Testing quadratic integration...\n";

  GaussLegendre gl;
  const auto quadratic = [](const double x) { return x * x; };

  // Integral of x^2 from -1 to 1 should be 2/3
  double expected = 2.0 / 3.0;

  double result2 = gl.eval<2>(quadratic);
  double result4 = gl.eval<4>(quadratic);
  double result8 = gl.eval<8>(quadratic);
  double result16 = gl.eval<16>(quadratic);
  double result32 = gl.eval<32>(quadratic);
  double result64 = gl.eval<64>(quadratic);

  assert(approx_equal(result2, expected));
  assert(approx_equal(result4, expected));
  assert(approx_equal(result8, expected));
  assert(approx_equal(result16, expected));
  assert(approx_equal(result32, expected));
  assert(approx_equal(result64, expected));

  std::cout << "✓ Quadratic integration tests passed\n";
}

void test_cubic_integration() {
  std::cout << "Testing cubic integration...\n";

  GaussLegendre gl;
  const auto cubic = [](const double x) { return x * x * x; };

  // Integral of x^3 from -1 to 1 should be 0 (odd function)
  double expected = 0.0;

  double result2 = gl.eval<2>(cubic);
  double result4 = gl.eval<4>(cubic);
  double result8 = gl.eval<8>(cubic);
  double result16 = gl.eval<16>(cubic);
  double result32 = gl.eval<32>(cubic);
  double result64 = gl.eval<64>(cubic);

  assert(approx_equal(result2, expected));
  assert(approx_equal(result4, expected));
  assert(approx_equal(result8, expected));
  assert(approx_equal(result16, expected));
  assert(approx_equal(result32, expected));
  assert(approx_equal(result64, expected));

  std::cout << "✓ Cubic integration tests passed\n";
}

void test_high_degree_polynomials() {
  std::cout << "Testing high degree polynomial integration...\n";

  GaussLegendre gl;

  // Test x^4: integral from -1 to 1 should be 2/5
  const auto poly4 = [](const double x) { return std::pow(x, 4); };
  double expected4 = 2.0 / 5.0;
  double result4_2pts = gl.eval<2>(poly4);
  double result4_4pts = gl.eval<4>(poly4);
  double result4_8pts = gl.eval<8>(poly4);

  // 2-point rule should be less accurate for x^4
  assert(!approx_equal(result4_2pts, expected4, 1e-10));
  // 4-point and higher should be exact for x^4
  assert(approx_equal(result4_4pts, expected4));
  assert(approx_equal(result4_8pts, expected4));

  // Test x^6: integral from -1 to 1 should be 2/7
  const auto poly6 = [](const double x) { return std::pow(x, 6); };
  double expected6 = 2.0 / 7.0;
  double result6_2pts = gl.eval<2>(poly6);
  double result6_4pts = gl.eval<4>(poly6);
  double result6_8pts = gl.eval<8>(poly6);

  std::cout << result6_4pts << expected6 << std::endl;

  // Lower order rules should be less accurate
  assert(!approx_equal(result6_2pts, expected6, 1e-6));
  assert(approx_equal(result6_4pts, expected6, 1e-8));
  // 8-point rule should be exact for x^6
  assert(approx_equal(result6_8pts, expected6));

  std::cout << "✓ High degree polynomial tests passed\n";
}

void test_transcendental_functions() {
  std::cout << "Testing transcendental function integration...\n";

  GaussLegendre gl;

  // Test sin(x) from -1 to 1: should be 0 (odd function)
  const auto sin_func = [](const double x) { return sin(x); };
  double sin_result = gl.eval<16>(sin_func);
  assert(approx_equal(sin_result, 0.0, 1e-10));

  // Test exp(x) from -1 to 1: should be e - 1/e ≈ 2.350402387
  const auto exp_func = [](const double x) { return exp(x); };
  double exp_expected = std::exp(1.0) - std::exp(-1.0);
  double exp_result16 = gl.eval<16>(exp_func);
  double exp_result32 = gl.eval<32>(exp_func);
  double exp_result64 = gl.eval<64>(exp_func);

  assert(approx_equal(exp_result16, exp_expected, 1e-8));
  assert(approx_equal(exp_result32, exp_expected, 1e-12));
  assert(approx_equal(exp_result64, exp_expected, 1e-14));

  std::cout << "✓ Transcendental function tests passed\n";
}

void test_convergence() {
  std::cout << "Testing convergence with increasing points...\n";

  GaussLegendre gl;
  const auto exp_func = [](const double x) { return exp(x); };
  double expected = std::exp(1.0) - std::exp(-1.0);

  double result2 = gl.eval<2>(exp_func);
  double result4 = gl.eval<4>(exp_func);
  double result8 = gl.eval<8>(exp_func);
  double result16 = gl.eval<16>(exp_func);
  double result32 = gl.eval<32>(exp_func);
  double result64 = gl.eval<64>(exp_func);

  double error2 = std::abs(result2 - expected);
  double error4 = std::abs(result4 - expected);
  double error8 = std::abs(result8 - expected);
  double error16 = std::abs(result16 - expected);
  double error32 = std::abs(result32 - expected);
  double error64 = std::abs(result64 - expected);

  // Errors should generally decrease as we increase points
  assert(error4 <= error2);
  assert(error8 <= error4);
  assert(error16 <= error8);
  assert(error32 <= error16);
  assert(error64 <= error32);

  std::cout << "✓ Convergence tests passed\n";
}

int main() {
  std::cout << "Running Gauss-Legendre Quadrature Unit Tests\n";
  std::cout << "============================================\n\n";

  try {
    test_constant_integration();
    test_linear_integration();
    test_quadratic_integration();
    test_cubic_integration();
    test_high_degree_polynomials();
    test_transcendental_functions();
    test_convergence();

    std::cout << "\n============================================\n";
    std::cout << "✅ All tests passed successfully!\n";
    std::cout << "============================================\n";

  } catch (const std::exception& e) {
    std::cout << "\n❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
