//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataSetBase...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/DataSetImpl.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

DataSetImpl::DataSetImpl ( hid_t id )
  : m_id( new hid_t(id), DataSetPtrDeleter() )
{
}

/// create new data set, specify the type explicitly
DataSetImpl
DataSetImpl::createDataSet ( hid_t parent,
                             const std::string& name,
                             const Type& type,
                             const DataSpace& dspc,
                             const PListDataSetCreate& plistDScreate,
                             const PListDataSetAccess& plistDSaccess )
{
  hid_t ds = H5Dcreate2 ( parent, name.c_str(), type.id(), dspc.id(),
                          H5P_DEFAULT, plistDScreate.plist(), plistDSaccess.plist() ) ;
  if ( ds < 0 ) throw Hdf5CallException( "DataSet::createDataSet", "H5Dcreate2" ) ;
  return DataSetImpl ( ds ) ;
}

/// open existing dataset
DataSetImpl
DataSetImpl::openDataSet ( hid_t parent, const std::string& name )
{
  hid_t ds = H5Dopen2 ( parent, name.c_str(), H5P_DEFAULT ) ;
  if ( ds < 0 ) throw Hdf5CallException( "DataSet::openDataSet", "H5Dopen2" ) ;
  return DataSetImpl ( ds ) ;
}

/// Changes the sizes of a dataset’s dimensions.
void
DataSetImpl::set_extent ( const hsize_t size[] )
{
  herr_t stat = H5Dset_extent( *m_id, size ) ;
  if ( stat < 0 ) throw Hdf5CallException( "DataSet::set_extent", "H5Dset_extent" ) ;
}


// store the data
void
DataSetImpl::store ( const Type& memType,
                     const DataSpace& memDspc,
                     const DataSpace& fileDspc,
                     const void* data )
{
  herr_t stat = H5Dwrite( *m_id, memType.id(), memDspc.id(), fileDspc.id(), H5P_DEFAULT, data ) ;
  if ( stat < 0 ) throw Hdf5CallException( "DataSet::store", "H5Dwrite" ) ;
}

// read the data
void 
DataSetImpl::read(const Type& memType,
                  const DataSpace& memDspc,
                  const DataSpace& fileDspc,
                  void* data)
{
  herr_t stat = H5Dread(*m_id, memType.id(), memDspc.id(), fileDspc.id(), H5P_DEFAULT, data);
  if ( stat < 0 ) throw Hdf5CallException( "DataSetImpl::read", "H5Dread" ) ;
}

/// access data space
DataSpace
DataSetImpl::dataSpace()
{
  hid_t dspc = H5Dget_space( *m_id ) ;
  if ( dspc < 0 ) throw Hdf5CallException( "DataSetImpl::dataSpace", "H5Dget_space" ) ;
  return DataSpace ( dspc ) ;
}


} // namespace hdf5pp
