#ifndef H5DATATYPES_H5DATAUTILS_H
#define H5DATATYPES_H5DATAUTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class H5DataUtils.
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
#include "hdf5pp/Group.h"

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
  void
  storeDataObject ( const T& data, const char* name, hdf5pp::Group grp )
  {
    // class must define persType() method returning its data type
    hdf5pp::Type type = T::persType() ;

    // make scalar data set
    hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, type, hdf5pp::DataSpace::makeScalar() ) ;

    // store data
    ds.store ( type, hdf5pp::DataSpace::makeScalar(), &data ) ;
  }

  template <typename T>
  void
  storeDataObjects ( hsize_t size, const T* data, hdf5pp::Type type, const char* name, hdf5pp::Group grp )
  {
    if ( size > 0 ) {
      // make simple data set
      hsize_t dims[1] = { size } ;
      hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeSimple ( 1, dims, dims ) ;
      hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, type, dsp ) ;
      // store data
      ds.store ( type, dsp, data ) ;
    } else {
      // for empty data set make null dataspace
      hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeNull () ;
      hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, type, dsp ) ;
      // store data
      ds.store ( type, dsp, data ) ;
    }
  }

  template <typename T>
  void
  storeDataObjects ( hsize_t size, const T* data, const char* name, hdf5pp::Group grp )
  {
    storeDataObjects ( size, data, T::persType(), name, grp );
  }

} // namespace H5DataTypes

#endif // H5DATATYPES_H5DATAUTILS_H
