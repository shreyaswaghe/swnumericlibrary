#pragma once

#include "TensorBase.hpp"

struct SingleBackend {
  //
  // ADD
  //
  template <typename TensorType>
  inline static void add(TensorBaseCRTP<TensorType>& lhs,
                         const TensorType::DataType& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] += rhs;
  }

  template <typename TensorType>
  inline static void add(TensorBaseCRTP<TensorType>& lhs,
                         const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    for (size_t i = 0; i < lhs.size(); i++) lhs[i] += rhs.sca * rhs.vec[i];
  }

  template <typename TensorType>
  inline static void add(TensorBaseCRTP<TensorType>& lhs,
                         const TensorBaseCRTP<TensorType>& rhs) {
    add(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                 .sca = typename TensorType::DataType(1.0), .vec = rhs});
  }

  //
  // SUBTRACT
  //
  template <typename TensorType>
  inline static void subtract(TensorBaseCRTP<TensorType>& lhs,
                              const TensorType::DataType& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] -= rhs;
  }

  template <typename TensorType>
  inline static void subtract(TensorBaseCRTP<TensorType>& lhs,
                              const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    for (size_t i = 0; i < lhs.size(); i++) lhs[i] -= rhs.sca * rhs.vec[i];
  }

  template <typename TensorType>
  inline static void subtract(TensorBaseCRTP<TensorType>& lhs,
                              const TensorBaseCRTP<TensorType>& rhs) {
    subtract(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                      .sca = typename TensorType::DataType(1.0), .vec = rhs});
  }

  //
  // MULTIPLY
  //
  template <typename TensorType>
  inline static void multiply(TensorBaseCRTP<TensorType>& lhs,
                              const TensorType::DataType& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs;
  }

  template <typename TensorType>
  inline static void multiply(TensorBaseCRTP<TensorType>& lhs,
                              const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs.sca * rhs.vec[i];
  }

  template <typename TensorType>
  inline static void multiply(TensorBaseCRTP<TensorType>& lhs,
                              const TensorBaseCRTP<TensorType>& rhs) {
    multiply(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                      .sca = typename TensorType::DataType(1.0), .vec = rhs});
  }

  //
  // DIVIDE
  //
  //
  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const TensorType::DataType& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs;
  }

  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs.sca * rhs.vec[i];
  }

  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const TensorBaseCRTP<TensorType>& rhs) {
    divide(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                    .sca = typename TensorType::DataType(1.0), .vec = rhs});
  }

  template <size_t _size, typename T>
  inline static void divide(Vector<_size, T, CblasBackend>& lhs, const T& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= T(1.0) / rhs;
  }

  //
  // ASSIGN
  //
  template <typename TensorType>
  inline static void assign(TensorBaseCRTP<TensorType>& lhs,
                            const TensorType::DataType& rhs) {
    std::fill_n(lhs.data(), lhs.size(), rhs);
  }

  template <typename TensorType>
  inline static void assign(TensorBaseCRTP<TensorType>& lhs,
                            const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.shape());
    };

    for (size_t i = 0; i < lhs.size(); i++) lhs[i] = rhs.sca * rhs.vec[i];
  }

  template <typename TensorType>
  inline static void assign(TensorBaseCRTP<TensorType>& lhs,
                            const TensorBaseCRTP<TensorType>& rhs) {
    assign(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                    .sca = TensorType::DataType(1.0), .vec = rhs});
  }

  //
  //  DOT
  //
  template <typename TensorType>
  inline static typename TensorType::DataType dot(
      const TensorBaseCRTP<TensorType>& lhs,
      const TensorBaseCRTP<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.shape());
    };

    typename TensorType::DataType acc = 0.0;
    for (size_t i = 0; i < lhs.size(); i++) acc += lhs[i] * rhs[i];
  }

  template <typename TensorType>
  inline static typename TensorType::DataType sdot(
      const TensorBaseCRTP<TensorType>& lhs,
      const TensorBaseCRTP<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.shape());
    };

    typename TensorType::DataType acc = 0.0;
    for (size_t i = 0; i < lhs.size(); i++) acc += lhs[i] * rhs[i];
  }

  //
  // NORMS
  //
  template <typename TensorType>
  inline static typename TensorType::DataType norm2(
      const TensorBaseCRTP<TensorType>& lhs) {
    typename TensorType::DataType acc = 0.0;
    for (size_t i = 0; i < lhs.size(); i++) acc += lhs[i] * lhs[i];
    return sqrt(acc);
  }

  template <typename TensorType>
  inline static typename TensorType::DataType norm1(
      const TensorBaseCRTP<TensorType>& lhs) {
    typename TensorType::DataType acc = 0.0;
    for (size_t i = 0; i < lhs.size(); i++) acc += std::abs(lhs[i]);
    return acc;
  }

  template <typename TensorType>
  inline static size_t norminf(const TensorBaseCRTP<TensorType>& lhs) {
    size_t maxIndex = indexmax(lhs);
    return std::abs(lhs[maxIndex]);
  }

  //
  // INDEXING
  //
  template <typename TensorType>
  inline static size_t indexmax(const TensorBaseCRTP<TensorType>& lhs) {
    size_t maxIndex = 0;
    typename TensorType::DataType maxElem = 0;
    for (size_t i = 0; i < lhs.size(); i++) {
      typename TensorType::DataType elem = lhs[i];
      if (std::abs(elem) > maxElem) {
        maxIndex = i;
        maxElem = elem;
      }
    }
    return maxIndex;
  }
};
