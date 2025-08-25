#pragma once

#include "TensorBase.hpp"
#include "cblas.h"

struct CblasBackend {
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

    if constexpr (std::is_same_v<typename TensorType::DataType, double>)
      cblas_daxpy(lhs.size(), rhs.sca, rhs.vec.data(), 1, lhs.data(), 1);
    else if constexpr (std::is_same_v<typename TensorType::DataType, float>)
      cblas_saxpy(lhs.size(), rhs.sca, rhs.vec.data(), 1, lhs.data(), 1);
    else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] += rhs.sca * rhs.vec[i];
    }
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

    if constexpr (std::is_same_v<typename TensorType::DataType, double>)
      cblas_daxpy(lhs.size(), -rhs.sca, rhs.vec.data(), 1, lhs.data(), 1);
    else if constexpr (std::is_same_v<typename TensorType::DataType, float>)
      cblas_saxpy(lhs.size(), -rhs.sca, rhs.vec.data(), 1, lhs.data(), 1);
    else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] -= rhs.sca * rhs.vec[i];
    }
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
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dscal(lhs.size(), rhs, lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_sscal(lhs.size(), rhs, lhs.data(), 1);
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs;
    }
  }

  template <typename TensorType>
  inline static void multiply(TensorBaseCRTP<TensorType>& lhs,
                              const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dscal(lhs.size(), rhs.sca, lhs.data(), 1);
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs.vec[i];
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_sscal(lhs.size(), rhs.sca, lhs.data(), 1);
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs.vec[i];
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs.sca * rhs.vec[i];
    }
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
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dscal(lhs.size(), TensorType::DataType(1.0) / rhs, lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_sscal(lhs.size(), TensorType::DataType(1.0) / rhs, lhs.data(), 1);
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= rhs;
    }
  }

  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const SCALAR_TIMES_TENSOR<TensorType>& rhs) {
    if constexpr (TensorTraits<TensorType>::_csize == 0) {
      assert(lhs.nDims() == rhs.vec.nDims());
      assert(lhs.shape() == rhs.vec.shape());
    };

    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dscal(lhs.size(), rhs.sca, lhs.data(), 1);
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs.vec[i];
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_sscal(lhs.size(), rhs.sca, lhs.data(), 1);
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs.vec[i];
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs.sca * rhs.vec[i];
    }
  }

  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const TensorBaseCRTP<TensorType>& rhs) {
    divide(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                    .sca = typename TensorType::DataType(1.0), .vec = rhs});
  }

  template <size_t _size, typename T>
  inline static void divide(Vector<_size, T, CblasBackend>& lhs, const T& rhs) {
    if constexpr (std::is_same_v<T, double>) {
      cblas_dscal(lhs.size(), 1.0 / rhs, lhs.data(), 1);
    } else if constexpr (std::is_same_v<T, float>) {
      cblas_sscal(lhs.size(), 1.0f / rhs, lhs.data(), 1);
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] *= T(1.0) / rhs;
    }
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

    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dcopy(lhs.size(), rhs.vec.data(), 1, lhs.data(), 1);
      cblas_dscal(lhs.size(), rhs.sca, lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_scopy(lhs.size(), rhs.vec.data(), 1, lhs.data(), 1);
      cblas_sscal(lhs.size(), rhs.sca, lhs.data(), 1);
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] = rhs.sca * rhs.vec[i];
    }
  }

  template <typename TensorType>
  inline static void assign(TensorBaseCRTP<TensorType>& lhs,
                            const TensorBaseCRTP<TensorType>& rhs) {
    assign(lhs, SCALAR_TIMES_TENSOR<TensorType>{
                    .sca = TensorType::DataType(1.0), .vec = rhs});
  }
};
