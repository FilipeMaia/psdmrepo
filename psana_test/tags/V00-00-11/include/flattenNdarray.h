#ifndef PSANA_TEST_FLATTENNDARRAY_H
#define PSANA_TEST_FLATTENNDARRAY_H

#include "ndarray/ndarray.h"
#include <vector>
#include <stdexcept>

namespace psana_test {

template <class T, unsigned R>
void flatten(const ndarray<const T,R> &a, std::vector<T> &v) {
  throw std::runtime_error("flatten( ndarray<const T,R>, vector<T> &) called for unsupported R value");
}

template <class T>
void flatten(const ndarray<const T,1> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0];
  v.resize(n);
  for (unsigned idx = 0; idx < n; idx ++ )  {
    v[idx] = a[idx];
  }
}

template <class T>
void flatten(const ndarray<const T,2> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0] * a.shape()[1];
  v.resize(n);
  unsigned idx = 0;
  for (unsigned i = 0; i < a.shape()[0]; ++i )  {
    for (unsigned j = 0; j < a.shape()[1]; ++j)  {
      v[idx++] = a[i][j];
    }
  }
}

template <class T>
void flatten(const ndarray<const T,3> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0] * a.shape()[1] * a.shape()[2];
  v.resize(n);
  unsigned idx = 0;
  for (unsigned i = 0; i < a.shape()[0]; ++i )  {
    for (unsigned j = 0; j < a.shape()[1]; ++j)  {
      for (unsigned k = 0; k < a.shape()[2]; ++k)  {
        v[idx++] = a[i][j][k];
      }
    }
  }
}

template <class T>
void flatten(const ndarray<const T,4> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3];
  v.resize(n);
  unsigned idx = 0;
  for (unsigned i = 0; i < a.shape()[0]; ++i )  {
    for (unsigned j = 0; j < a.shape()[1]; ++j)  {
      for (unsigned k = 0; k < a.shape()[2]; ++k)  {
        for (unsigned l = 0; l < a.shape()[3]; ++l)  {
          v[idx++] = a[i][j][k][l];
        }
      }
    }
  }
}

template <class T>
void flatten(const ndarray<const T,5> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] * a.shape()[4];
  v.resize(n);
  unsigned idx = 0;
  for (unsigned i = 0; i < a.shape()[0]; ++i )  {
    for (unsigned j = 0; j < a.shape()[1]; ++j)  {
      for (unsigned k = 0; k < a.shape()[2]; ++k)  {
        for (unsigned l = 0; l < a.shape()[3]; ++l)  {
          for (unsigned m = 0; m < a.shape()[4]; ++m)  {
            v[idx++] = a[i][j][k][l][m];
          }
        }
      }
    }
  }
}

template <class T>
void flatten(const ndarray<const T,6> &a, std::vector<T> &v) {
  unsigned n = a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] * a.shape()[4] * a.shape()[5];
  v.resize(n);
  unsigned idx = 0;
  for (unsigned i = 0; i < a.shape()[0]; ++i )  {
    for (unsigned j = 0; j < a.shape()[1]; ++j)  {
      for (unsigned k = 0; k < a.shape()[2]; ++k)  {
        for (unsigned l = 0; l < a.shape()[3]; ++l)  {
          for (unsigned m = 0; m < a.shape()[4]; ++m)  {
            for (unsigned n = 0; n < a.shape()[5]; ++n)  {
              v[idx++] = a[i][j][k][l][m][n];
            }
          }
        }
      }
    }
  }
}

}; // namespace psana_test

#endif
