#ifndef HDF5PP_ARRAYTYPE_H
#define HDF5PP_ARRAYTYPE_H

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
#include "hdf5/hdf5.h"
#include "hdf5pp/Type.h"
#include "hdf5pp/TypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  HDF5 type interface
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: Type.h 250 2009-04-08 01:02:05Z salnikov $
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

/**
 *  Class for the array types, supports operations applicable to array types only
 */
class ArrayType : public Type {
public:

  // make an array type of rank 1
  static ArrayType arrayType( const Type& baseType, hsize_t dim ) {
    return arrayType( baseType, 1, &dim ) ;
  }

  // make an array type of rank 1
  template <typename T>
  static ArrayType arrayType( hsize_t dim ) {
    return arrayType( TypeTraits<T>::native_type(), 1, &dim ) ;
  }

  // make an array type of any rank
  static ArrayType arrayType( const Type& baseType, unsigned rank, hsize_t dims[] ) ;

protected:

  ArrayType ( hid_t id ) : Type( id, true ) {}

private:

};

} // namespace hdf5pp

#endif // HDF5PP_ARRAYTYPE_H
