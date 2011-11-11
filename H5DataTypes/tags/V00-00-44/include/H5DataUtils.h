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
  hdf5pp::DataSet<T>
  storeDataObject ( const T& data, const char* name, hdf5pp::Group grp )
  {
    // make scalar data set
    hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, hdf5pp::DataSpace::makeScalar() ) ;

    // store data
    ds.store ( hdf5pp::DataSpace::makeScalar(), &data ) ;
    
    return ds;
  }

  template <typename T>
  hdf5pp::DataSet<T>
  storeDataObjects ( hsize_t size, const T* data, const char* name, hdf5pp::Group grp )
  {
    if ( size > 0 ) {
      // make simple data set
      hsize_t dims[1] = { size } ;
      hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeSimple ( 1, dims, dims ) ;
      hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, dsp ) ;
      // store data
      ds.store ( dsp, data ) ;
      return ds;
    } else {
      // for empty data set make null dataspace
      hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeNull () ;
      hdf5pp::DataSet<T> ds = grp.createDataSet<T> ( name, dsp ) ;
      // store data
      ds.store ( dsp, data ) ;
      return ds;
    }
  }

} // namespace H5DataTypes

#endif // H5DATATYPES_H5DATAUTILS_H
