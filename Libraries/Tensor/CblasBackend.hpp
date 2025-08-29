#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>

#include "TensorBase.hpp"
#include "cblas.h"

namespace swnumeric {

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
                      .sca = typename TensorType::DataType(1), .vec = rhs});
  }

  //
  // DIVIDE
  //
  template <typename TensorType>
  inline static void divide(TensorBaseCRTP<TensorType>& lhs,
                            const TensorType::DataType& rhs) {
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      cblas_dscal(lhs.size(), typename TensorType::DataType(1.0) / rhs,
                  lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      cblas_sscal(lhs.size(), typename TensorType::DataType(1.0) / rhs,
                  lhs.data(), 1);
    } else {
      for (size_t i = 0; i < lhs.size(); i++) lhs[i] /= rhs;
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
                    .sca = typename TensorType::DataType(1), .vec = rhs});
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

    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      return cblas_ddot(lhs.size(), lhs.data(), 1, rhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      return cblas_sdot(lhs.size(), lhs.data(), 1, rhs.data(), 1);
    } else {
      typename TensorType::DataType acc = 0.0;
      for (size_t i = 0; i < lhs.size(); i++) acc += lhs[i] * rhs[i];
    }
  }

  //
  // NORMS
  //
  template <typename TensorType>
  inline static typename TensorType::DataType norm2(
      const TensorBaseCRTP<TensorType>& lhs) {
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      return cblas_dnrm2(lhs.size(), lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      return cblas_snrm2(lhs.size(), lhs.data(), 1);
    } else {
      typename TensorType::DataType acc = 0.0;
      for (size_t i = 0; i < lhs.size(); i++) acc += lhs[i] * lhs[i];
      return sqrt(acc);
    }
  }

  template <typename TensorType>
  inline static typename TensorType::DataType norm1(
      const TensorBaseCRTP<TensorType>& lhs) {
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      return cblas_dasum(lhs.size(), lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      return cblas_sasum(lhs.size(), lhs.data(), 1);
    } else {
      typename TensorType::DataType acc = 0;
      for (size_t i = 0; i < lhs.size(); i++) acc += std::abs(lhs[i]);
      return acc;
    }
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
    if constexpr (std::is_same_v<typename TensorType::DataType, double>) {
      return cblas_idamax(lhs.size(), lhs.data(), 1);
    } else if constexpr (std::is_same_v<typename TensorType::DataType, float>) {
      return cblas_isamax(lhs.size(), lhs.data(), 1);
    } else {
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
  }
};

}  // namespace swnumeric
