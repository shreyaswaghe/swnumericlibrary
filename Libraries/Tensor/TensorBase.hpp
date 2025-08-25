#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>

// Some Backends to support
struct CblasBackend;
struct SingleBackend;
struct CUDABackend;

#define DEFAULT_BACKEND CblasBackend
#define DEFAULT_DATATYPE double
#define DEFAULT_SIZE 0

// Basic Type Definitions
template <size_t ssize = DEFAULT_SIZE, typename T = DEFAULT_DATATYPE,
          typename Backend = DEFAULT_BACKEND>
struct Vector;

template <size_t rows, size_t cols, typename T = DEFAULT_DATATYPE,
          typename Backend = DEFAULT_BACKEND>
struct Matrix;

// Each TensorType must specialize this to provide information about the type
// which should be known/infered at compile time:
// 1. TensorTraits::DataType representing the data type of an ordinary element
// of TensorType
// 2. TensorTraits::BackendType representing the Backend interface
// 3. TensorTraits::_nDims representing the number of dimensions of the
// TensorType
// 4. TensorTraits::_csize representing the number of elements that may be
// deduced at compile time
template <typename TensorType>
struct TensorTraits;

// CRTP Base Class for TensorType
// Defines the minimum interface that is expected of a TensorType
template <typename DerivedType>
struct TensorBaseCRTP {
  using Traits = TensorTraits<DerivedType>;
  using DataType = typename Traits::DataType;

  DerivedType& derived() { return static_cast<DerivedType&>(*this); }
  const DerivedType& derived() const {
    return static_cast<const DerivedType&>(*this);
  }

  DataType* data() { return derived().data(); }
  const DataType* data() const { return derived().data(); }

  DataType* operator()(size_t i) { return derived()(i); }
  const DataType* operator()(size_t i) const { return derived()(i); }

  DataType& operator[](size_t i) { return derived()[i]; }
  const DataType& operator[](size_t i) const { return derived()[i]; }

  size_t size() const { return derived().size(); }
  std::array<size_t, Traits::_nDims> shape() const { return derived().shape(); }
  size_t nDims() const { return Traits::_nDims; }
};

// Helper Implementation Data Aggregate
template <typename TensorType>
struct SCALAR_TIMES_TENSOR {
  const TensorType::DataType sca;
  const TensorBaseCRTP<TensorType>& vec;
};

// TODO: Specialize TensorTypes to use this memory alignment for
// TensorType.data()
constexpr size_t alignmentByteSize = 32;
