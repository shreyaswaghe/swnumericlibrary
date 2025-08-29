#pragma once

#include <algorithm>
#include <cstddef>

#include "CblasBackend.hpp"
#include "TensorBase.hpp"

namespace swnumeric {

//
// Matrix specialization of TensorType, compatible with TensorBaseCRTP
//
template <size_t nrows, size_t ncols, typename T, typename Backend>
struct TensorTraits<Matrix<nrows, ncols, T, Backend>> {
  using DataType = T;
  using BackendType = Backend;
  static constexpr size_t _nDims = 2;
  static constexpr size_t _csize = nrows * ncols;
  static constexpr size_t _nrows = nrows;
  static constexpr size_t _ncols = ncols;
};

template <size_t nrows, size_t ncols, typename T, typename Backend>
struct Matrix : TensorBaseCRTP<Matrix<nrows, ncols, T, Backend>> {
 public:
  using DataType = TensorTraits<Matrix>::DataType;
  union {
    DataType *heapData;
    DataType staticData[nrows * ncols > 0 ? nrows *ncols : 1];
  } dataHolder = {nullptr};
  size_t _rows = 0, _cols = 0;

  void alloc(size_t nr, size_t nc) {
    void *mem = std::malloc(nr * nc * sizeof(DataType));
    assert(mem != nullptr);
    dataHolder.heapData = static_cast<DataType *>(mem);
    _rows = nr;
    _cols = nc;
  }

  bool isAlloc() const {
    if constexpr (nrows * ncols == 0)
      return dataHolder.heapData != nullptr;
    else
      return true;
  }

  // Construction with memory allocation
  Matrix(size_t nr = nrows, size_t nc = ncols) {
    if constexpr (nrows * ncols == 0) {
      if (nr * nc == 0) return;
      alloc(nr, nc);
    } else {
      _rows = nrows;
      _cols = ncols;
    }
  }

  Matrix(const std::array<size_t, 2> &shape) {
    size_t nr = shape[0];
    size_t nc = shape[1];
    if constexpr (nrows * ncols == 0) {
      if (nr * nc == 0) return;
      alloc(nr, nc);
    } else {
      _rows = nrows;
      _cols = ncols;
    }
  }

  Matrix(std::initializer_list<std::initializer_list<DataType>> init)
      : Matrix(init.size(), init.size() > 0 ? init.begin()->size() : 0) {
    size_t i = 0;
    for (auto rowIt = init.begin(); rowIt != init.end(); ++rowIt, ++i) {
      size_t j = 0;
      for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt, ++j) {
        (*this)[idx(i, j)] = *colIt;
      }
    }
  }
  Matrix(const std::array<std::array<DataType, ncols>, nrows> &vals)
      : Matrix(nrows, ncols) {
    for (size_t i = 0; i < nrows; i++) {
      for (size_t j = 0; j < ncols; j++) {
        (*this)[idx(i, j)] = vals[i][j];
      }
    }
  }

  // Copy construction
  Matrix(const Matrix &other) {
    _rows = other._rows;
    _cols = other._cols;
    if constexpr (nrows * ncols == 0) {
      if (_rows * _cols > 0) {
        dataHolder.heapData = static_cast<DataType *>(
            std::malloc(_rows * _cols * sizeof(DataType)));
        std::copy(other.data(), other.data() + _rows * _cols,
                  dataHolder.heapData);
      }
    } else {
      std::copy(other.data(), other.data() + _rows * _cols,
                dataHolder.staticData);
    }
  }

  // Copy assignment
  Matrix &operator=(const Matrix &other) {
    if (this == &other) return *this;
    if constexpr (nrows * ncols == 0) {
      assert(_rows == other._rows && _cols == other._cols);
      if (dataHolder.heapData) {
        std::free(dataHolder.heapData);
        dataHolder.heapData = nullptr;
      }
      _rows = other._rows;
      _cols = other._cols;
      if (_rows * _cols > 0) {
        dataHolder.heapData = static_cast<DataType *>(
            std::malloc(_rows * _cols * sizeof(DataType)));
        std::copy(other.data(), other.data() + _rows * _cols,
                  dataHolder.heapData);
      }
    } else {
      _rows = other._rows;
      _cols = other._cols;
      std::copy(other.data(), other.data() + _rows * _cols,
                dataHolder.staticData);
    }
    return *this;
  }

  // Move constructor
  Matrix(Matrix &&other) noexcept {
    _rows = other._rows;
    _cols = other._cols;
    if constexpr (nrows * ncols == 0) {
      dataHolder.heapData = other.dataHolder.heapData;
      other.dataHolder.heapData = nullptr;
      other._rows = 0;
      other._cols = 0;
    } else {
      std::copy(other.data(), other.data() + _rows * _cols,
                dataHolder.staticData);
    }
  }

  // Move assignment
  Matrix &operator=(Matrix &&other) noexcept {
    if (this == &other) return *this;
    if constexpr (nrows * ncols == 0) {
      if (dataHolder.heapData) {
        std::free(dataHolder.heapData);
        dataHolder.heapData = nullptr;
      }
      dataHolder.heapData = other.dataHolder.heapData;
      _rows = other._rows;
      _cols = other._cols;
      other.dataHolder.heapData = nullptr;
      other._size = 0;
    } else {
      _rows = other._rows;
      _cols = other._cols;
      std::copy(other.data(), other.data() + _rows * _cols,
                dataHolder.staticData);
    }
    return *this;
  }

  // Destructor
  ~Matrix() {
    if (nrows * ncols == 0) {
      std::free(dataHolder.heapData);
      dataHolder.heapData = nullptr;
    }
  }

  //
  // Simple Accessors
  //

  inline DataType *data() {
    if constexpr (nrows * ncols == 0)
      return dataHolder.heapData;
    else
      return dataHolder.staticData;
  }

  inline const DataType *data() const {
    if constexpr (nrows * ncols == 0)
      return dataHolder.heapData;
    else
      return dataHolder.staticData;
  }

  inline const DataType &operator[](size_t i) const { return data()[i]; }
  inline DataType &operator[](size_t i) { return data()[i]; }

  inline const DataType *operator()(size_t i) const { return data() + i; }
  inline DataType *operator()(size_t i) { return data() + i; }
  inline DataType operator()(size_t i, size_t j) { return data()[idx(i, j)]; }

  //
  // Shape Information Getters
  //

  inline const size_t size() const { return _rows * _cols; }
  inline const std::array<size_t, 2> shape() const { return {_rows, _cols}; }
  inline const size_t nDims() const { return TensorTraits<Matrix>::_nDims; }
  inline const size_t lda() const { return _rows; }
  inline const size_t rows() const { return _rows; }
  inline const size_t cols() const { return _cols; }
  inline const size_t idx(size_t r, size_t c) { return r + c * lda(); }

  //
  // Backend handles specialization of operators
  //

  template <typename Expr>
  inline Matrix &operator+=(const Expr &other) {
    Backend::add(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Matrix &operator-=(const Expr &other) {
    Backend::subtract(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Matrix &operator*=(const Expr &other) {
    Backend::multiply(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Matrix &operator/=(const Expr &other) {
    Backend::divide(*this, other);
    return *this;
  }

  template <typename Expr>
  inline Matrix &operator=(const Expr &other) {
    Backend::assign(*this, other);
    return *this;
  }

  inline DataType dot(const Matrix &other) const {
    return Backend::dot(*this, other);
  }

  inline DataType normfro() const { return Backend::norm2(*this); }
  inline DataType norm2() const { return Backend::norm2(*this); }
  inline DataType norm1() const { return Backend::norm1(*this); }
  inline DataType norminf() const { return Backend::norminf(*this); }

  inline size_t indexmax() const { return Backend::indexmax(*this); }
};

//
// For Matrix, it can be convenient to not do just in-place operations
//

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator+(
    const Matrix<nrows, ncols, T, Backend> &lhs,
    const Matrix<nrows, ncols, T, Backend> &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res += other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator+(
    const Matrix<nrows, ncols, T, Backend> &lhs, const T &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res += other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator-(
    Matrix<nrows, ncols, T, Backend> &lhs,
    const Matrix<nrows, ncols, T, Backend> &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res -= other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator-(
    const Matrix<nrows, ncols, T, Backend> &lhs, const T &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res -= other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator*(
    const Matrix<nrows, ncols, T, Backend> &lhs,
    const Matrix<nrows, ncols, T, Backend> &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res *= other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Matrix<nrows, ncols, T, Backend>> operator*(
    const Matrix<nrows, ncols, T, Backend> &lhs, const T &other) {
  return SCALAR_TIMES_TENSOR{.sca = other, .vec = lhs};
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Matrix<nrows, ncols, T, Backend>> operator*(
    const T &other, const Matrix<nrows, ncols, T, Backend> &lhs) {
  return SCALAR_TIMES_TENSOR<Matrix<nrows, ncols, T, Backend>>{.sca = other,
                                                               .vec = lhs};
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline Matrix<nrows, ncols, T, Backend> operator/(
    const Matrix<nrows, ncols, T, Backend> &lhs,
    const Matrix<nrows, ncols, T, Backend> &other) {
  Matrix<nrows, ncols, T, Backend> res(lhs.shape());
  res = lhs;
  res /= other;
  return res;
}

template <size_t nrows, size_t ncols, typename T, typename Backend>
inline SCALAR_TIMES_TENSOR<Matrix<nrows, ncols, T, Backend>> operator/(
    const Matrix<nrows, ncols, T, Backend> &lhs, const T &other) {
  return SCALAR_TIMES_TENSOR{.sca = T(1.0) / other, .vec = lhs};
}

}  // namespace swnumeric
