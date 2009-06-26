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

protected:

private:

  // Data members
  PListImpl m_impl ;
};

} // namespace hdf5pp

#endif // HDF5PP_PLISTFILECREATE_H
