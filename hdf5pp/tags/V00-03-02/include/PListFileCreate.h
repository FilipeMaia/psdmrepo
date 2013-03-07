#ifndef HDF5PP_PLISTFILECREATE_H
#define HDF5PP_PLISTFILECREATE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListFileCreate.
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
#include "hdf5pp/PListImpl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Property list for file creation
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

namespace hdf5pp {

class PListFileCreate  {
public:

  // Default constructor
  PListFileCreate () ;

  // Destructor
  ~PListFileCreate () ;

  // accessor
  hid_t plist() const { return m_impl.id() ; }

  // set the node size for chunked datasets b-tree,
  // see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetIstoreK
  void set_istore_k(unsigned ik) ;
  
  // set the parameters for symbols b-tree,
  // see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetSymK
  void set_sym_k(unsigned ik, unsigned lk) ;
  
  // Sets user block size, see
  // http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetUserblock
  void set_userblock(hsize_t size);
  
  // Sets the byte size of the offsets and lengths in an HDF5 file,
  // see http://www.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-SetSizes
  void set_sizes(size_t sizeof_addr, size_t sizeof_size); 

protected:

private:

  // Data members
  PListImpl m_impl ;
};

} // namespace hdf5pp

#endif // HDF5PP_PLISTFILECREATE_H
