// Copyright (c) 2025 Shreyas Waghe. All Rights Reserved.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "CblasBackend.hpp"
#include "TensorBase.hpp"

//
// Vector specialization of TensorType, compatible with TensorBaseCRTP
//
template <size_t ssize, typename T, typename Backend>
struct TensorTraits<Vector<ssize, T, Backend>> {
  using DataType = T;
  using BackendType = Backend;
  static constexpr size_t _nDims = 1;
  static constexpr size_t _csize = ssize;
};

template <size_t ssize, typename T, typename Backend>
struct Vector : TensorBaseCRTP<Vector<ssize, T, Backend>> {
 public:
  using DataType = TensorTraits<Vector>::DataType;
  static constexpr size_t _csize = ssize;
  union {
    DataType *heapData;
    DataType staticData[ssize > 0 ? ssize : 1];
  } dataHolder;
  size_t _size;

  // Construction with memory allocation
  Vector(size_t sz = ssize) {
    if constexpr (ssize == 0) {
      assert(sz != 0);
      void *mem = std::malloc(sz * sizeof(DataType));
      assert(mem != nullptr);
      dataHolder.heapData = static_cast<DataType *>(mem);
      _size = sz;
    } else {
      _size = ssize;
    }
  }

  // Construction from array-like objects
  // Disable construction from initializer list since its size cannot be
  // obtained at compile time in C++20
  Vector(const std::array<T, ssize> &vals) : Vector(ssize) {
    for (size_t i = 0; i < ssize; i++) (*this)[i] = vals[i];
  }

  // Copy construction
  Vector(const Vector &other) {
    _size = other._size;
    if constexpr (ssize == 0) {
      if (_size > 0) {
        dataHolder.heapData =
            static_cast<DataType *>(std::malloc(_size * sizeof(DataType)));
        std::copy(other.data(), other.data() + _size, dataHolder.heapData);
      }
    } else {
      std::copy(other.data(), other.data() + _size, dataHolder.staticData);
    }
  }

  // Copy assignment
  Vector &operator=(const Vector &other) {
    if (this == &other) return *this;
    if constexpr (ssize == 0) {
      if (dataHolder.heapData) std::free(dataHolder.heapData);
      _size = other._size;
      if (_size > 0) {
        dataHolder.heapData =
            static_cast<DataType *>(std::malloc(_size * sizeof(DataType)));
        std::copy(other.data(), other.data() + _size, dataHolder.heapData);
      }
    } else {
      _size = other._size;
      std::copy(other.data(), other.data() + _size, dataHolder.staticData);
    }
    return *this;
  }

  // Move constructor
  Vector(Vector &&other) noexcept {
    _size = other._size;
    if constexpr (ssize == 0) {
      dataHolder.heapData = other.dataHolder.heapData;
      other.dataHolder.heapData = nullptr;
      other._size = 0;
    } else {
      std::copy(other.data(), other.data() + _size, dataHolder.staticData);
    }
  }

  // Move assignment
  Vector &operator=(Vector &&other) noexcept {
    if (this == &other) return *this;
    if constexpr (ssize == 0) {
      if (dataHolder.heapData) std::free(dataHolder.heapData);
      dataHolder.heapData = other.dataHolder.heapData;
      _size = other._size;
      other.dataHolder.heapData = nullptr;
      other._size = 0;
    } else {
      _size = other._size;
      std::copy(other.data(), other.data() + _size, dataHolder.staticData);
    }
    return *this;
  }

  // Destructor
  ~Vector() {
    if (ssize == 0) {
      std::free(dataHolder.heapData);
      dataHolder.heapData = nullptr;
    }
  }

  //
  // Simple Accessors
  //

  inline DataType *data() {
    if constexpr (ssize == 0)
      return dataHolder.heapData;
    else
      return dataHolder.staticData;
  }

  inline const DataType *data() const {
    if constexpr (ssize == 0)
      return dataHolder.heapData;
    else
      return dataHolder.staticData;
  }

  inline const DataType &operator[](size_t i) const { return data()[i]; }
  inline DataType &operator[](size_t i) { return data()[i]; }

  inline const DataType *operator()(size_t i) const { return data() + i; }
  inline DataType *operator()(size_t i) { return data() + i; }

  //
  // Shape Information Getters
  //

  inline const size_t size() const { return _size; }
  inline const std::array<size_t, 1> shape() const { return {_size}; }
  inline const size_t nDims() const { return TensorTraits<Vector>::_nDims; }

  //
  // Backend handles specialization of operators
  //

  template <typename Expr>
  inline Vector &operator+=(const Expr &other) {
    Backend::add(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Vector &operator-=(const Expr &other) {
    Backend::subtract(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Vector &operator*=(const Expr &other) {
    Backend::multiply(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Vector &operator/=(const Expr &other) {
    Backend::divide(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Vector &operator=(const Expr &other) {
    Backend::assign(*this, other);
    return *this;
  }
};

//
// For Vector, it can be convenient to not do just in-place operations
//

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator+(
    const Vector<ssize, T, Backend> &lhs,
    const Vector<ssize, T, Backend> &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res += other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator+(const Vector<ssize, T, Backend> &lhs,
                                           const T &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res += other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator-(
    Vector<ssize, T, Backend> &lhs, const Vector<ssize, T, Backend> &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res -= other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator-(const Vector<ssize, T, Backend> &lhs,
                                           const T &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res -= other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator*(
    const Vector<ssize, T, Backend> &lhs,
    const Vector<ssize, T, Backend> &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res *= other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Vector<ssize, T, Backend>> operator*(
    const Vector<ssize, T, Backend> &lhs, const T &other) {
  return SCALAR_TIMES_TENSOR{.sca = other, .vec = lhs};
}

template <size_t ssize, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Vector<ssize, T, Backend>> operator*(
    const T &other, const Vector<ssize, T, Backend> &lhs) {
  return SCALAR_TIMES_TENSOR<Vector<ssize, T, Backend>>{.sca = other,
                                                        .vec = lhs};
}

template <size_t ssize, typename T, typename Backend>
inline Vector<ssize, T, Backend> operator/(
    const Vector<ssize, T, Backend> &lhs,
    const Vector<ssize, T, Backend> &other) {
  Vector<ssize, T, Backend> res(lhs.size());
  res = lhs;
  res /= other;
  return res;
}

template <size_t ssize, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Vector<ssize, T, Backend>> operator/(
    const Vector<ssize, T, Backend> &lhs, const T &other) {
  return SCALAR_TIMES_TENSOR{.sca = T(1.0) / other, .vec = lhs};
}
