#ifndef HDF5PP_COMPOUNDTYPE_H
#define HDF5PP_COMPOUNDTYPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: Type.h 250 2009-04-08 01:02:05Z salnikov $
//
// Description:
//	Class Type.
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
#include "hdf5pp/Type.h"
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
 *  Class for the compound types, supports operations applicable to compound types only
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: Type.h 250 2009-04-08 01:02:05Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class CompoundType : public Type {
public:

  // make a compound type
  template <typename T>
  static CompoundType compoundType() { return compoundType( sizeof(T) ) ;  }

  static CompoundType compoundType( size_t size ) ;

  // add one more member
  void insert ( const char* name, size_t offset, const Type& t, size_t size=0 ) ;

  template <typename U>
  void insert_native ( const char* name, size_t offset, size_t size=0 ) {
    return insert ( name, offset, TypeTraits<U>::native_type(size) ) ;
  }

  template <typename U>
  void insert_stored ( const char* name, size_t offset, size_t size=0 ) {
    return insert ( name, offset, TypeTraits<U>::stored_type(size) ) ;
  }

protected:

  CompoundType ( hid_t id ) : Type( id, true ) {}

private:

};

} // namespace hdf5pp

#endif // HDF5PP_COMPOUNDTYPE_H
