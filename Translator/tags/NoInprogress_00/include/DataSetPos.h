#ifndef TRANSLATOR_DSETPOS_H
#define TRANSLATOR_DSETPOS_H

#include "hdf5/hdf5.h"

namespace Translator {

/**
 * @brief Keeps track of a dataset and position within dataset.
 *
 * Keeps track of the size of the dataset based on calls to increaseByOne().
 * Actual hdf5 dataset size can differ if calls to increaseByOne() are not kept in 
 * sync with hdf5 write calls.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider   
 */
class DataSetPos {
public:
  typedef enum {Unlimited, Fixed} Shape;
  DataSetPos() {};
 DataSetPos(hid_t dsetId, Shape shape=Unlimited) : 
  m_dsetId(dsetId), m_currentSize(0), m_shape(shape) {};
  hid_t dsetId() const { return m_dsetId; }               /// the hdf5 dataset id
  hsize_t currentSize() const { return m_currentSize; }   /// the current size of the dataset, based on accumulated 
                                                          /// calls to increaseSizeByOne()
  void increaseSizeByOne() { ++m_currentSize; }     /// increases recorded size for dataset
  Shape shape() const { return m_shape; }    
 private:
  hid_t m_dsetId;
  hsize_t m_currentSize;
  Shape m_shape;
};

} // namespace
#endif
