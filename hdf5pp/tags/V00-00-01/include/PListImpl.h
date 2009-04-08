#ifndef HDF5PP_PLISTIMPL_H
#define HDF5PP_PLISTIMPL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListImpl.
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
#include "hdf5/hdf5.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Property list wrapper
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

class PListImpl  {
public:

  // Default constructor
  PListImpl ( hid_t cls ) ;

  // Destructor
  ~PListImpl () ;

  PListImpl ( const PListImpl& ) ;

  PListImpl& operator = ( const PListImpl& ) ;

  hid_t id() const { return m_id ; }

protected:

private:

  // Data members
  hid_t m_id ;
};

} // namespace hdf5pp

#endif // HDF5PP_PLISTIMPL_H
