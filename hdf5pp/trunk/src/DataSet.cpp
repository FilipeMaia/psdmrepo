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
#include "hdf5pp/DataSet.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  char logger[] = "hdf5pp.DataSet";

  // deleter for  boost smart pointer
  struct DataSetPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) {
        MsgLog(logger, debug, "DataSetPtrDeleter: dataset=" << *id) ;
        H5Dclose ( *id );
      }
      delete id ;
    }
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

DataSet::DataSet(hid_t id)
  : m_id( new hid_t(id), DataSetPtrDeleter() )
{
  MsgLog(logger, debug, "DataSet ctor: " << id) ;
}

/// create new data set, specify the type explicitly
DataSet
DataSet::createDataSet ( hid_t parent,
                             const std::string& name,
                             const Type& type,
                             const DataSpace& dspc,
                             const PListDataSetCreate& plistDScreate,
                             const PListDataSetAccess& plistDSaccess )
{
  MsgLog(logger, debug, "DataSet::createDataSet: name=" << name << " parent=" << parent) ;
  hid_t ds = H5Dcreate2 ( parent, name.c_str(), type.id(), dspc.id(),
                          H5P_DEFAULT, plistDScreate.plist(), plistDSaccess.plist() ) ;
  if ( ds < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dcreate2" ) ;
  return DataSet ( ds ) ;
}

/// open existing dataset
DataSet
DataSet::openDataSet ( hid_t parent, const std::string& name,
    const PListDataSetAccess& plistDSaccess)
{
  MsgLog(logger, debug, "DataSet::openDataSet: name=" << name << " parent=" << parent) ;
  hid_t ds = H5Dopen2 ( parent, name.c_str(), plistDSaccess.plist() ) ;
  if ( ds < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dopen2" ) ;
  return DataSet ( ds ) ;
}

/// Changes the sizes of a dataset’s dimensions.
void
DataSet::set_extent(const hsize_t size[])
{
  herr_t stat = H5Dset_extent( *m_id, size ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dset_extent" ) ;
}


// store the data
void
DataSet::_store(const Type& memType,
                const DataSpace& memDspc,
                const DataSpace& fileDspc,
                const void* data)
{
  herr_t stat = H5Dwrite( *m_id, memType.id(), memDspc.id(), fileDspc.id(), H5P_DEFAULT, data ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dwrite" ) ;
}

// read the data
void 
DataSet::_read(const Type& memType,
               const DataSpace& memDspc,
               const DataSpace& fileDspc,
               void* data)
{
  herr_t stat = H5Dread(*m_id, memType.id(), memDspc.id(), fileDspc.id(), H5P_DEFAULT, data);
  if ( stat < 0 ) {
    MsgLog(logger, debug, "H5Dread failed, memtype = " << memType);
    throw Hdf5CallException( ERR_LOC, "H5Dread" ) ;
  }
}

// reclaim space allocated to vlen structures
void
DataSet::_vlen_reclaim(const hdf5pp::Type& type, const DataSpace& memDspc, void* data)
{
  herr_t stat = H5Dvlen_reclaim(type.id(), memDspc.id(), H5P_DEFAULT, data);
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dvlen_reclaim" ) ;
}

/// access data space
DataSpace
DataSet::dataSpace()
{
  hid_t dspc = H5Dget_space( *m_id ) ;
  if ( dspc < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dget_space" ) ;
  return DataSpace ( dspc ) ;
}

/// get chunk size, this method only works for 1-dim datasets
size_t
DataSet::chunkSize() const
{
  hid_t plist = H5Dget_create_plist(*m_id);
  if (plist < 0) throw Hdf5CallException( ERR_LOC, "H5Dget_create_plist" ) ;
  hsize_t dims[1];
  int nd = H5Pget_chunk(plist, 1, dims);
  H5Pclose(plist);
  if (nd < 0) throw Hdf5CallException( ERR_LOC, "H5Pget_chunk" ) ;
  return dims[0];
}

/// access dataset type
Type
DataSet::type()
{
  hid_t typeId = H5Dget_type( *m_id ) ;
  if ( typeId < 0 ) throw Hdf5CallException( ERR_LOC, "H5Dget_type" ) ;
  return Type::UnlockedType(typeId) ;
}

// get group name (absolute)
std::string 
DataSet::name() const
{
  const int maxsize = 255;
  char buf[maxsize+1];

  // first try with the fixed buffer size
  ssize_t size = H5Iget_name(*m_id, buf, maxsize+1);
  if (size < 0) {
    throw Hdf5CallException( ERR_LOC, "H5Iget_name") ;
  }
  if (size == 0) {
    // name is not known
    return std::string();
  }
  if (size <= maxsize) {
    // name has fit into buffer
    return buf;
  }

  // another try with dynamically allocated buffer
  char* dbuf = new char[size+1];
  H5Iget_name(*m_id, dbuf, size+1);
  std::string res(dbuf);
  delete [] dbuf;
  return res;
}

} // namespace hdf5pp
