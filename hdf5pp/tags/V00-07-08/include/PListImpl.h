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

namespace hdf5pp {

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  Property list wrapper
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class PListImpl  {
public:

  // Default constructor
  PListImpl();
  
  // Counstructor which accepts property list class
  PListImpl ( hid_t cls ) ;

  // Destructor
  ~PListImpl () ;

  PListImpl ( const PListImpl& ) ;

  PListImpl& operator = ( const PListImpl& ) ;

  hid_t id() const { return m_id ; }
  
  bool isDefault() const { return m_id == H5P_DEFAULT; }

  // changes class, only if id is H5P_DEFAULT
  void setClass(hid_t cls);

protected:

private:

  // Data members
  hid_t m_id ;
};

} // namespace hdf5pp

#endif // HDF5PP_PLISTIMPL_H
