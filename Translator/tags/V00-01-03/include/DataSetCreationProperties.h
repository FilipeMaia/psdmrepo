#ifndef TRANSLATOR_DATASETCREATIONPROPERTIES_H
#define TRANSLATOR_DATASETCREATIONPROPERTIES_H

#include "hdf5/hdf5.h"
#include "Translator/ChunkPolicy.h"

namespace Translator {

/**
 * @brief class to store hdf5 dataset creation properties.
 *
 * Holds three attributes to be used when making a hdf5 dataset creation property list:
 *
 * ChunkPolicy - an object that returns the chunk size and chunk cache size based on 
 *         the type of data in the dataset (depends primarily on the size of the data)
 * shuffle - boolean, true if bytes in dataset  should be shuffled to help compression
 * deflate - int, if 0-9, apply that level of compression, -1 means no gzip compression
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider   
 */
class DataSetCreationProperties {
 public:


  // constructors
  DataSetCreationProperties() 
    : m_shuffle(false), m_deflate(0) {};

  DataSetCreationProperties(boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy,
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
  boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy() const { return m_chunkPolicy; }
  bool shuffle() const { return m_shuffle; }
  int deflate() const { return m_deflate; }

 private:
  boost::shared_ptr<Translator::ChunkPolicy> m_chunkPolicy;
  bool m_shuffle;
  int m_deflate;
};

} // namespace


#endif
