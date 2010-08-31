//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListFileCreate...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/PListFileCreate.h"

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

//----------------
// Constructors --
//----------------
PListFileCreate::PListFileCreate ()
  : m_impl(H5P_FILE_CREATE)
{
}

//--------------
// Destructor --
//--------------
PListFileCreate::~PListFileCreate ()
{
}

// set the node size for chunked datasets b-tree,
// see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetIstoreK
void 
PListFileCreate::set_istore_k(unsigned ik) 
{
  herr_t stat = H5Pset_istore_k ( m_impl.id(), ik ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( "PListFileCreate::set_istore_k", "H5Pset_istore_k" ) ;
  }
}

// set the parameters for symbols b-tree,
// see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetSymK
void 
PListFileCreate::set_sym_k(unsigned ik, unsigned lk) 
{
  herr_t stat = H5Pset_sym_k ( m_impl.id(), ik, lk ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( "PListFileCreate::set_sym_k", "H5Pset_sym_k" ) ;
  }
}

// Sets user block size, see
// http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetUserblock
void 
PListFileCreate::set_userblock(hsize_t size)
{
  herr_t stat = H5Pset_userblock ( m_impl.id(), size ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( "PListFileCreate::set_userblock", "H5Pset_userblock" ) ;
  }
}

// Sets the byte size of the offsets and lengths in an HDF5 file,
// see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetSizes
void 
PListFileCreate::set_sizes(size_t sizeof_addr, size_t sizeof_size)
{
  herr_t stat = H5Pset_sizes ( m_impl.id(), sizeof_addr, sizeof_size ) ;
  if ( stat < 0 ) {
    throw Hdf5CallException ( "PListFileCreate::set_sizes", "H5Pset_sizes" ) ;
  }
}

} // namespace hdf5pp
