#ifndef H5DATATYPES_OBJECTCONTAINER_H
#define H5DATATYPES_OBJECTCONTAINER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ObjectContainer.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/DataSet.h"
#include "hdf5pp/DataSpace.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/TypeTraits.h"
#include "hdf5pp/Type.h"
#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

template <typename T>
class ObjectContainer {
public:

  typedef T value_type;

  /// Provide type at run time
  ObjectContainer ( const std::string& name,
                    hdf5pp::Group location,
                    const hdf5pp::Type& stored_type,
                    hsize_t chunk_size,
                    int deflate,
                    bool shuffle)
    : m_dataset()
    , m_count(0)
  {
    // make extensible data space
    hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeSimple ( 0, H5S_UNLIMITED ) ;

    // use chunking
    hdf5pp::PListDataSetCreate plDScreate ;
    plDScreate.set_chunk(chunk_size) ;
    if ( shuffle ) plDScreate.set_shuffle() ;
    if ( deflate >= 0 ) plDScreate.set_deflate(deflate) ;

    // adjust chunk cache size to fit at least one chunk
    hdf5pp::PListDataSetAccess plDSaccess;
    const hsize_t def_chunk_cache_size = 1024*1024;
    const hsize_t max_chunk_cache_size = 10*1024*1024;
    hsize_t chunk_cache_size = chunk_size * stored_type.size();
    if (chunk_cache_size < def_chunk_cache_size) {
      chunk_cache_size = def_chunk_cache_size;
    } else  if (chunk_cache_size < max_chunk_cache_size) {
      chunk_cache_size *= 2;
    }
    hsize_t nchunks_in_cache = chunk_cache_size / (chunk_size * stored_type.size());
    MsgLog("ObjectContainer", trace, "ObjectContainer -- set chunk cache size to " << chunk_cache_size) ;
    plDSaccess.set_chunk_cache(nchunks_in_cache*20, chunk_cache_size);

    // make a data set
    MsgLog("ObjectContainer", trace, "ObjectContainer -- creating dataset " << name << " with chunk size " << chunk_size << " (objects)") ;
    if (location.hasChild(name)) {
      m_dataset = location.openDataSet(name);
    } else {
      m_dataset = location.createDataSet(name, stored_type, dsp, plDScreate, plDSaccess);
    }
  }

  /**
   *  Append one more data element to the end of data set
   */
  void append ( const T& obj, const hdf5pp::Type& native_type = hdf5pp::TypeTraits<T>::native_type() )
  {
    // extend data set
    unsigned long newCount = m_count + 1 ;
    m_dataset.set_extent ( newCount ) ;

    // define hyperslab for file data set
    hdf5pp::DataSpace fileDspc = m_dataset.dataSpace() ;
    hsize_t start[] = { m_count } ;
    hsize_t size[] = { 1 } ;
    fileDspc.select_hyperslab ( H5S_SELECT_SET, start, 0, size, 0 ) ;

    // define in-memory data space
    hdf5pp::DataSpace memDspc = hdf5pp::DataSpace::makeScalar() ;

    // store it
    m_dataset.store ( memDspc, fileDspc, &obj, native_type ) ;

    m_count = newCount ;
  }

  /// get current size of this container
  unsigned long size() const { return m_count; }

  /// set container size, fill with defaults if necessary
  void resize(unsigned long newSize) {
    m_dataset.set_extent(newSize);
    m_count = newSize;
  }

  /// get access to data set
  hdf5pp::DataSet& dataset() { return m_dataset ; }

private:

  hdf5pp::DataSet m_dataset ;
  unsigned long m_count ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_OBJECTCONTAINER_H
