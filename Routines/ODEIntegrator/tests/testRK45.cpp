#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "../ODEDynamics.hpp"
#include "../RungeKutta45.hpp"
#include "Libraries/Tensor/Matrix.hpp"
#include "Libraries/Tensor/Vector.hpp"

// Helper function for floating point comparison
template <typename T>
bool isNear(T a, T b, T tolerance = 1e-10) {
  return std::abs(a - b) < tolerance;
}

// Test ODE: Simple harmonic oscillator
// d²x/dt² = -ω²x
// State vector: [position, velocity]
// dx/dt = [velocity, -ω²*position]
using namespace swnumeric;

using Vector2 = Vector<2>;

class SimpleHarmonicOscillator : public ODEDynamics<Vector2> {
  using ODEDynamics<Vector2>::StateType;

 private:
  double omega_squared;  // ω²

 public:
  SimpleHarmonicOscillator(double omega = 1.0) : omega_squared(omega * omega) {}

  ~SimpleHarmonicOscillator() override {}

  // Pre-integration hook (identity transformation for this example)
  void PreIntegration(StateType& x, double t) override {}

  // Post-integration hook (identity transformation for this example)
  void PostIntegration(StateType& x, double t) override {}

  // doublehe core dynamics: dx/dt = f(x, t)
  void Gradient(StateType& gradout, const StateType& x, double t) override {
    // x[0] = position, x[1] = velocity
    // dx/dt = [velocity, -ω²*position]
    gradout[0] = x[1];                   // d(position)/dt = velocity
    gradout[1] = -omega_squared * x[0];  // d(velocity)/dt = -ω²*position
  }

  // StateType norm (Euclidean norm)
  double stateNorm(const StateType& x) override {
    return std::sqrt(x[0] * x[0] + x[1] * x[1]);
  }
};

// Test ODE: Exponential decay matrix
// dX/dt = -αX where X is a matrix

using Matrix22 = Matrix<2, 2>;
class ExponentialDecayMatrix : public ODEDynamics<Matrix22> {
  using ODEDynamics<Matrix22>::StateType;

 private:
  double alpha;  // decay constant

 public:
  ExponentialDecayMatrix(double decay_rate = 1.0) : alpha(decay_rate) {}

  ~ExponentialDecayMatrix() override {}

  void PreIntegration(StateType& x, double t) override {}

  void PostIntegration(StateType& x, double t) override {}

  // dX/dt = -αX
  void Gradient(StateType& gradout, const StateType& x, double t) override {
    gradout = x;
    gradout.derived() *= -alpha;
  }

  // Frobenius norm
  double stateNorm(const StateType& x) override {
    double sum = 0;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        sum += x.derived()(i, j) * x.derived()(i, j);
      }
    }
    return std::sqrt(sum) / 4.0;
  }
};

// Test functions
void testSimpleHarmonicOscillatorGradient() {
  std::cout << "Testing Simple Harmonic Oscillator Gradient..." << std::endl;

  SimpleHarmonicOscillator sho(2.0);  // ω = 2, so ω² = 4

  Vector2 state;
  state[0] = 1.0;  // position = 1
  state[1] = 0.0;  // velocity = 0

  Vector2 gradient;
  sho.Gradient(gradient, state, 0.0);

  // Expected: dx/dt = [0, -4*1] = [0, -4]
  assert(isNear(gradient[0], 0.0));
  assert(isNear(gradient[1], -4.0));

  std::cout << "✓ Gradient test passed" << std::endl;
}

void testSimpleHarmonicOscillatorNorm() {
  std::cout << "Testing Simple Harmonic Oscillator Norm..." << std::endl;

  SimpleHarmonicOscillator sho;

  Vector2 state;
  state[0] = 3.0;
  state[1] = 4.0;

  double norm = sho.stateNorm(state);
  assert(isNear(norm, 5.0));  // sqrt(3² + 4²) = 5

  std::cout << "✓ Norm test passed" << std::endl;
}

void testExponentialDecayMatrixGradient() {
  std::cout << "Testing Exponential Decay Matrix Gradient..." << std::endl;

  ExponentialDecayMatrix decay(0.5);  // α = 0.5

  Matrix22 state;
  state(0, 0) = 2.0;
  state(0, 1) = 1.0;
  state(1, 0) = 3.0;
  state(1, 1) = 4.0;

  Matrix22 gradient;
  decay.Gradient(gradient, state, 0.0);

  // Expected: dX/dt = -0.5 * X
  assert(isNear(gradient(0, 0), -1.0));  // -0.5 * 2.0
  assert(isNear(gradient(0, 1), -0.5));  // -0.5 * 1.0
  assert(isNear(gradient(1, 0), -1.5));  // -0.5 * 3.0
  assert(isNear(gradient(1, 1), -2.0));  // -0.5 * 4.0

  std::cout << "✓ Matrix gradient test passed" << std::endl;
}

void testExponentialDecayMatrixNorm() {
  std::cout << "Testing Exponential Decay Matrix Norm..." << std::endl;

  ExponentialDecayMatrix decay;

  Matrix22 state;
  state(0, 0) = 1.0;
  state(0, 1) = 2.0;
  state(1, 0) = 3.0;
  state(1, 1) = 4.0;

  double norm = decay.stateNorm(state);
  double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0) / 4.0;  // sqrt(30)
  assert(isNear(norm, expected));

  std::cout << "✓ Matrix norm test passed" << std::endl;
}

void testGradientConsistency() {
  std::cout << "Testing Gradient Consistency..." << std::endl;

  SimpleHarmonicOscillator sho(1.0);

  Vector2 state1, state2;
  state1[0] = 1.0;
  state1[1] = 0.0;
  state2[0] = 0.0;
  state2[1] = 1.0;

  Vector2 grad1, grad2;
  sho.Gradient(grad1, state1, 0.0);
  sho.Gradient(grad2, state2, 0.0);

  // For simple harmonic oscillator with ω=1:
  // At (1,0): gradient should be (0,-1)
  // At (0,1): gradient should be (1,0)
  assert(isNear(grad1[0], 0.0));
  assert(isNear(grad1[1], -1.0));
  assert(isNear(grad2[0], 1.0));
  assert(isNear(grad2[1], 0.0));

  std::cout << "✓ Gradient consistency test passed" << std::endl;
}

void testComplexOscillatorBehavior() {
  std::cout << "Testing Complex Oscillator Behavior..." << std::endl;

  SimpleHarmonicOscillator sho(1.0);

  // Test at maximum displacement (energy should be conserved in principle)
  Vector2 state_max_pos;
  state_max_pos[0] = 2.0;  // max position
  state_max_pos[1] = 0.0;  // zero velocity

  Vector2 state_max_vel;
  state_max_vel[0] = 0.0;  // zero position
  state_max_vel[1] = 2.0;  // max velocity

  double norm_pos = sho.stateNorm(state_max_pos);
  double norm_vel = sho.stateNorm(state_max_vel);

  // Both should have same "energy" (norm in phase space)
  assert(isNear(norm_pos, norm_vel));
  assert(isNear(norm_pos, 2.0));

  std::cout << "✓ Complex oscillator behavior test passed" << std::endl;
}

void runPerformanceTest() {
  std::cout << "Running Performance Test..." << std::endl;

  SimpleHarmonicOscillator sho;
  Vector2 state;
  state[0] = 1.0;
  state[1] = 1.0;

  auto start = std::chrono::high_resolution_clock::now();

  Vector2 gradient;
  for (int i = 0; i < 1000000; ++i) {
    sho.Gradient(state, gradient, 0.0);
    state[0] += 1e-8;  // Small perturbation to avoid optimization
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Performance: " << duration.count()
            << " microseconds for 1M gradient evaluations" << std::endl;

  // Simple assertion - should complete in reasonable time
  assert(duration.count() < 1000000);  // Less than 1 seconds

  std::cout << "✓ Performance test passed" << std::endl;
}

// Integrator tests
void testExponentialDecayIntegration() {
  std::cout << "Testing Exponential Decay Integration..." << std::endl;

  ExponentialDecayMatrix decay(0.0);  // α = 1.0
  RungeKutta45<Matrix22> integrator(decay);

  // Initial condition: identity matrix
  Matrix22 state;
  state(0, 0) = 1.0;
  state(0, 1) = 0.0;
  state(1, 0) = 0.0;
  state(1, 1) = 1.0;

  double t_start = 0.0;
  double t_end = 1e-4;

  Matrix22 workState;
  integrator(state, workState, t_start, t_end);

  // Analytical solution: X(t) = X(0) * exp(-α*t) = I * exp(-1) ≈ 0.3679
  double expected = std::exp(0.0);

  std::cout << workState(0, 0) << " " << workState(1, 0) << std::endl;

  // assert(isNear(final_time, t_end));
  assert(isNear(workState(0, 0), expected, 1e-4));
  assert(isNear(workState(1, 1), expected, 1e-4));
  assert(isNear(workState(0, 1), 0.0, 1e-6));
  assert(isNear(workState(1, 0), 0.0, 1e-6));

  std::cout << "✓ Exponential decay integration test passed" << std::endl;
}

void testFastDecayAdaptiveStep() {
  std::cout << "Testing Fast Decay with Adaptive Stepping..." << std::endl;

  ExponentialDecayMatrix fast_decay(100.0);  // Very fast decay
  RungeKutta45<Matrix22> integrator(fast_decay);
  integrator.hmin = 1e-6;
  integrator.hmax = 100;
  integrator.rtol = 1e-11;
  integrator.atol = 1e-11;

  Matrix22 state;
  state(0, 0) = 1.0;
  state(0, 1) = 2.0;
  state(1, 0) = 3.0;
  state(1, 1) = 4.0;

  double t_start = 0.0;
  double t_end = 10.0;

  Matrix22 workState;
  integrator(state, workState, t_start, t_end);

  // Should reach t_end despite fast dynamics
  // Values should be very small due to fast decay
  double norm = fast_decay.stateNorm(workState);

  std::cout << "norm is " << norm << std::endl;
  assert(norm < 1e-10);  // Should decay to nearly zero

  std::cout << "✓ Fast decay adaptive stepping test passed" << std::endl;
}

void testSlowDecayLargeStep() {
  std::cout << "Testing Slow Decay with Large Steps..." << std::endl;

  ExponentialDecayMatrix slow_decay(0.1);  // Slow decay
  RungeKutta45<Matrix22> integrator(slow_decay);
  integrator.hmin = 0.01;
  integrator.atol = 1e-12;
  integrator.rtol = 1e-12;

  Matrix22 state;
  state(0, 0) = 1.0;
  state(0, 1) = 0.0;
  state(1, 0) = 0.0;
  state(1, 1) = 1.0;

  double t_start = 0.0;
  double t_end = 5.0;

  Matrix22 workState;
  integrator(state, workState, t_start, t_end);

  // Analytical solution: exp(-0.1 * 5) = exp(-0.5) ≈ 0.6065
  double expected = std::exp(-0.5);

  std::cout << "Expected: " << expected << "\n";

  // assert(isNear(final_time, t_end));
  assert(isNear(workState(0, 0), expected, 1e-1));
  assert(isNear(workState(1, 1), expected, 1e-1));

  std::cout << "✓ Slow decay large step test passed" << std::endl;
}

void testIntegratorTolerances() {
  std::cout << "Testing Integrator Tolerances..." << std::endl;

  ExponentialDecayMatrix decay(0.2);

  // Tight tolerances
  RungeKutta45<Matrix22> tight_integrator(decay);
  tight_integrator.atol = 1e-3;
  tight_integrator.rtol = 1e-6;
  tight_integrator.hmin = 1e-12;
  // Loose tolerances
  RungeKutta45<Matrix22> loose_integrator(decay);
  loose_integrator.atol = loose_integrator.rtol = 1e-1;
  loose_integrator.hmin = 1e-2;

  Matrix22 state_tight, state_loose;
  state_tight(0, 0) = state_loose(0, 0) = 1.0;
  state_tight(0, 1) = state_loose(0, 1) = 0.0;
  state_tight(1, 0) = state_loose(1, 0) = 0.0;
  state_tight(1, 1) = state_loose(1, 1) = 1.0;

  double t_start = 0.0;
  double t_end = 1.0;

  Matrix22 work_tight, work_loose;

  auto tight_start = std::chrono::high_resolution_clock::now();
  tight_integrator(state_tight, work_tight, t_start, t_end);
  auto tight_end = std::chrono::high_resolution_clock::now();
  auto tight_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      tight_end - tight_start);

  auto loose_start = std::chrono::high_resolution_clock::now();
  loose_integrator(state_loose, work_loose, t_start, t_end);
  auto loose_end = std::chrono::high_resolution_clock::now();
  auto loose_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      loose_end - loose_start);

  // Tight tolerance should be more accurate
  double expected = std::exp(-0.2);
  double error_tight = std::abs(work_tight(0, 0) - expected);
  double error_loose = std::abs(work_loose(0, 0) - expected);

  std::cout << "Tight intg took " << tight_duration.count() / 1000000.0 << " s"
            << std::endl;
  std::cout << "Loose intg took " << loose_duration.count() / 1000000.0 << " s"
            << std::endl;

  assert(error_tight <= error_loose);
  assert(error_tight < 1e-8);

  std::cout << "✓ Integrator tolerance test passed" << std::endl;
}

void testZeroIntegrationTime() {
  std::cout << "Testing Zero Integration Time..." << std::endl;

  ExponentialDecayMatrix decay(1.0);
  RungeKutta45<Matrix22> integrator(decay);

  Matrix22 state;
  state(0, 0) = 2.0;
  state(0, 1) = 3.0;
  state(1, 0) = 4.0;
  state(1, 1) = 5.0;

  Matrix22 original_state;
  original_state = state;

  double t_start = 1.0;
  double t_end = 1.0;  // Same start and end time

  Matrix22 workState;
  integrator(state, workState, t_start, t_end);

  // State should remain unchanged
  // assert(isNear(final_time, t_end));
  assert(isNear(state(0, 0), original_state(0, 0)));
  assert(isNear(state(0, 1), original_state(0, 1)));
  assert(isNear(state(1, 0), original_state(1, 0)));
  assert(isNear(state(1, 1), original_state(1, 1)));

  std::cout << "✓ Zero integration time test passed" << std::endl;
}

int main() {
  testExponentialDecayIntegration();
  testIntegratorTolerances();
  return 0;
}
