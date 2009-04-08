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
#include "hdf5pp/Type.h"

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

  ObjectContainer ( const char* name,
                    hdf5pp::Group& location,
                    const hdf5pp::Type& type = T::persType(),
                    hsize_t chunk_size = 10240,
                    int deflate = -1 )
    : m_dataset()
    , m_type( type )
    , m_count(0)
  {
    // make extensible data space
    hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeSimple ( 0, H5S_UNLIMITED ) ;

    // use chunking
    hdf5pp::PListDataSetCreate plDScreate ;
    plDScreate.set_chunk(chunk_size) ;
    if ( deflate >= 0 ) plDScreate.set_deflate(deflate) ;

    // make a data set
    m_dataset = location.createDataSet<T> ( name, m_type, dsp, plDScreate ) ;
  }

  void append ( const T& obj )
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
    m_dataset.store ( m_type, memDspc, fileDspc, &obj ) ;

    m_count = newCount ;
  }

private:

  hdf5pp::DataSet<T> m_dataset ;
  hdf5pp::Type m_type ;
  unsigned long m_count ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_OBJECTCONTAINER_H
