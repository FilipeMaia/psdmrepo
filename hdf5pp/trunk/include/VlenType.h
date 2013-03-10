#ifndef HDF5PP_VLENTYPE_H
#define HDF5PP_VLENTYPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class VlenType.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "hdf5pp/Type.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/TypeTraits.h"

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
 *  HDF5 variable length type interface
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class VlenType : public Type {
public:

  // make a variable-length array type with base type defined in the argument
  static VlenType vlenType( const Type& baseType ) ;

protected:

  VlenType ( hid_t id ) : Type( id, true ) {}

private:

};

} // namespace hdf5pp

#endif // HDF5PP_VLENTYPE_H
