#ifndef TRANSLATOR_DSETPOS_H
#define TRANSLATOR_DSETPOS_H

#include "hdf5/hdf5.h"

namespace Translator {

class DataSetPos {
public:
  /**
   * @brief Keeps track of a dataset and position within dataset.
   *
   * Keeps track of the size of the dataset based on calls to increaseByOne().
   * Actual hdf5 dataset size can differ if calls to increaseByOne() are not kept in 
   * sync with hdf5 write calls.
   */
  typedef enum {Unlimited, Fixed} MaxSize;
  DataSetPos() {};
 DataSetPos(hid_t dsetId, MaxSize maxSize=Unlimited) : m_dsetId(dsetId), m_currentSize(0), m_maxSize(maxSize) {};
  hid_t dsetId() { return m_dsetId; }               /// the hdf5 dataset id
  hsize_t currentSize() { return m_currentSize; }   /// the current size of the dataset, based on accumulated 
                                                    /// calls to increaseSizeByOne()
  void increaseSizeByOne() { ++m_currentSize; }     /// increases recorded size for dataset
  MaxSize maxSize() { return m_maxSize; }           /// wether or not this is Unlimited size, or Fixed
 private:
  hid_t m_dsetId;
  hsize_t m_currentSize;
  MaxSize m_maxSize;
};

} // namespace
#endif
