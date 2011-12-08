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

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

template <typename T>
class ObjectContainer {
public:

  /// Provide type at run time
  ObjectContainer ( const std::string& name,
                    hdf5pp::Group& location,
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
    hsize_t objectsPerChunk = numObjectsPerChunk(stored_type, chunk_size);
    plDScreate.set_chunk(objectsPerChunk) ;
    if ( shuffle ) plDScreate.set_shuffle() ;
    if ( deflate >= 0 ) plDScreate.set_deflate(deflate) ;

    // adjust chunk cache size to fit at least one chunk
    hdf5pp::PListDataSetAccess plDSaccess;
    const hsize_t def_chunk_cache_size = 1024*1024;
    const hsize_t max_chunk_cache_size = 10*1024*1024;
    hsize_t chunk_cache_size = objectsPerChunk * stored_type.size();
    if (chunk_cache_size < def_chunk_cache_size) {
      chunk_cache_size = def_chunk_cache_size;
    } else  if (chunk_cache_size < max_chunk_cache_size) {
      chunk_cache_size *= 2;
    }
    hsize_t nchunks_in_cache = chunk_cache_size / (objectsPerChunk * stored_type.size());
    MsgLog("ObjectContainer", trace, "ObjectContainer -- set chunk cache size to " << chunk_cache_size) ;
    plDSaccess.set_chunk_cache(nchunks_in_cache*20, chunk_cache_size);

    // make a data set
    MsgLog("ObjectContainer", trace, "ObjectContainer -- creating dataset " << name << " with chunk size " << objectsPerChunk << " (objects)") ;
    if (location.hasChild(name)) {
      m_dataset = location.openDataSet<T> ( name ) ;
    } else {
      m_dataset = location.createDataSet<T> ( name, stored_type, dsp, plDScreate, plDSaccess ) ;
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

  /// get access to data set
  hdf5pp::DataSet<T>& dataset() { return m_dataset ; }

private:

  /// Calculate suitable chunk size from input data
  static hsize_t numObjectsPerChunk (const hdf5pp::Type& stored_type, hsize_t chunk_size)
  {
    const hsize_t abs_max_chunk_size = 100*1024*1024;
    const unsigned max_objects_per_chunk = 2048;
    const unsigned min_objects_per_chunk = 50;

    // chunk_size is a desirable size, make sure that number of objects is not
    // too large or too small
    size_t obj_size = stored_type.size();
    hsize_t chunk = chunk_size / obj_size;
    if (chunk > max_objects_per_chunk) {
      chunk = max_objects_per_chunk;
    } else if (chunk < min_objects_per_chunk) {
      chunk = min_objects_per_chunk;
      if (chunk*obj_size > abs_max_chunk_size) chunk = abs_max_chunk_size / obj_size;
    }
    return chunk;
  }

  hdf5pp::DataSet<T> m_dataset ;
  unsigned long m_count ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_OBJECTCONTAINER_H
