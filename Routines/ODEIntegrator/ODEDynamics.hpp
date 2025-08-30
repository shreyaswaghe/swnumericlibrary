#pragma once

#include <cstddef>

#include "Libraries/Tensor/TensorBase.hpp"

namespace swnumeric {

// Abstract class defining minimum interface for ODE Integration
template <typename StateTypee>
struct ODEDynamics {
  using StateType = StateTypee;
  using StateCRTP = TensorBaseCRTP<StateType>;

  virtual ~ODEDynamics() = 0;
  virtual void PreIntegration(StateType& x, double t) = 0;
  virtual void PostIntegration(StateType& x, double t) = 0;
  virtual void Gradient(StateType& gradOut, const StateType& x, double t) = 0;
  virtual double stateNorm(const StateType& x) = 0;
};

template <typename StateType>
ODEDynamics<StateType>::~ODEDynamics(){};

};  // namespace swnumeric
