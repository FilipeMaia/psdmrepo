#ifndef TRANSLATOR_DATASETCREATIONPROPERTIES_H
#define TRANSLATOR_DATASETCREATIONPROPERTIES_H

#include "hdf5/hdf5.h"
#include "psddl_hdf2psana/ChunkPolicy.h"

namespace Translator {

class DataSetCreationProperties {
 public:

  /**
   * @brief class to store dfh5 dataset creation properties.
   *
   * Holds three attributes to be used when making a hdf5 dataset creation property list:
   *
   * ChunkPolicy - an object that returns the chunk size and chunk cache size
   *               based on the type of data in the dataset (depends primarily on the size of the data)
   * shuffle - boolean, true if bytes in dataset  should be shuffled to help compression
   * deflate - int, if 1-9, apply that level of compression
   */

  // constructors
  DataSetCreationProperties() 
    : m_shuffle(false), m_deflate(0) {};

  DataSetCreationProperties(boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> chunkPolicy,
                            bool shuffle, int deflate) 
    : m_chunkPolicy(chunkPolicy), m_shuffle(shuffle), m_deflate(deflate) {};

  DataSetCreationProperties(const DataSetCreationProperties & other) 
    : m_chunkPolicy(other.chunkPolicy()), 
    m_shuffle(other.shuffle()), m_deflate(other.deflate()) {};

  // assignment operator
  DataSetCreationProperties & operator=(const DataSetCreationProperties &other) {
    m_chunkPolicy = other.chunkPolicy();
    m_shuffle = other.shuffle();
    m_deflate = other.deflate();
    return *this;
  }  

  // accessors
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> chunkPolicy() const { return m_chunkPolicy; }
  bool shuffle() const { return m_shuffle; }
  int deflate() const { return m_deflate; }

 private:
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_chunkPolicy;
  bool m_shuffle;
  int m_deflate;
};

} // namespace


#endif
