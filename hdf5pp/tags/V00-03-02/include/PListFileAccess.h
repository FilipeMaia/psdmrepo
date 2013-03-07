#ifndef HDF5PP_PLISTFILEACCESS_H
#define HDF5PP_PLISTFILEACCESS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListFileAccess.
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
 *  Property list for file access
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

class PListFileAccess  {
public:

  // Default constructor
  PListFileAccess () ;

  // Destructor
  ~PListFileAccess () ;

  // accessor
  hid_t plist() const { return m_impl.id() ; }

  // use family driver
  void set_family_driver ( hsize_t memb_size, const PListFileAccess& memb_fapl ) ;

protected:

private:

  // Data members
  PListImpl m_impl ;

};

} // namespace hdf5pp

#endif // HDF5PP_PLISTFILEACCESS_H
