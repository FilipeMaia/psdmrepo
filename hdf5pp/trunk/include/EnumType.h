#ifndef HDF5PP_ENUMTYPE_H
#define HDF5PP_ENUMTYPE_H

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
#include "hdf5pp/Type.h"

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"
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
 *  Class for the enum types, supports operations applicable to enum types only
 */
template <typename T>
class EnumType : public Type {
public:

  // make an enum type based on some integer type
  static EnumType enumType() {
    hid_t tid = H5Tenum_create( TypeTraits<T>::native_type().id() ) ;
    if ( tid < 0 ) throw Hdf5CallException ( "EnumType::enumType", "H5Tenum_create" ) ;
    return EnumType ( tid ) ;
  }

  void insert ( const char* name, T value ) {
    herr_t stat = H5Tenum_insert( id(), name, static_cast<void *>(&value) ) ;
    if ( stat < 0 ) throw Hdf5CallException ( "EnumType::insert", "H5Tenum_insert" ) ;
  }

protected:

  EnumType ( hid_t id ) : Type( id, true ) {}

private:

};

} // namespace hdf5pp

#endif // HDF5PP_ENUMTYPE_H
