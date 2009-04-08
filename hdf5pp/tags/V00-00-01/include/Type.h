#ifndef HDF5PP_TYPE_H
#define HDF5PP_TYPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Type.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

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
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

class Type {
public:

  // create locked type from type id
  static Type LockedType( hid_t tid ) { return Type(tid,false); }
  // create unlocked type from type id
  static Type UnlockedType( hid_t tid ) { return Type(tid,true); }

  // Default constructor
  Type() ;

  // Destructor
  ~Type () ;

  /// return type id
  hid_t id() const { return *m_id ; }

protected:

  // constructor
  Type ( hid_t id, bool doClose ) ;

private:

  // deleter for  boost smart pointer
  struct TypePtrDeleter {
    TypePtrDeleter(bool doClose ) : m_doClose(doClose) {}
    void operator()( hid_t* id ) {
      if ( id and m_doClose ) H5Tclose ( *id );
      delete id ;
    }
    bool m_doClose ;
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

/**
 *  Class for the atomic types, supports operations applicable to atomic types only
 */
class AtomicType : public Type {
public:

  // make a non-modifiable atomic type
  template <typename T>
  static AtomicType atomicType() {
    return AtomicType ( TypeTraits<T>::h5type_native(), false ) ;
  }

  // make a modifiable atomic type
  template <typename T>
  static AtomicType atomicTypeCopy() {
    hid_t tid = H5Tcopy ( TypeTraits<T>::h5type_native() ) ;
    if ( tid < 0 ) throw Hdf5CallException ( "AtomicType::atomicTypeCopy", "H5Tcopy" ) ;
    return AtomicType ( tid, true ) ;
  }


protected:

  AtomicType ( hid_t id, bool doClose ) : Type( id, doClose ) {}

private:

};

/**
 *  Class for the compound types, supports operations applicable to compound types only
 */
class CompoundType : public Type {
public:

  // make a compound type
  template <typename T>
  static CompoundType compoundType() { return compoundType( sizeof(T) ) ;  }

  static CompoundType compoundType( size_t size ) ;

  // add one more member
  void insert ( const char* name, size_t offset, Type t ) ;

protected:

  CompoundType ( hid_t id ) : Type( id, true ) {}

private:

};

/**
 *  Class for the array types, supports operations applicable to array types only
 */
class ArrayType : public Type {
public:

  // make an array type of rank 1
  template <typename T>
  static ArrayType arrayType( hsize_t dim ) {
    return arrayType( AtomicType::atomicType<T>(), dim ) ;
  }

  // make an array type of any rank
  template <typename T>
  static ArrayType arrayType( unsigned rank, hsize_t dims[] ) {
    return arrayType( AtomicType::atomicType<T>(), rank, dims ) ;
  }

  // make an array type of rank 1
  static ArrayType arrayType( Type baseType, hsize_t dim ) {
    hsize_t dims[1] = { dim };
    return arrayType( baseType, 1, dims ) ;
  }

  // make an array type of any rank
  static ArrayType arrayType( Type baseType, unsigned rank, hsize_t dims[] ) ;

protected:

  ArrayType ( hid_t id ) : Type( id, true ) {}

private:

};

/**
 *  Class for the enum types, supports operations applicable to enum types only
 */
template <typename T>
class EnumType : public Type {
public:

  // make an enum type based on some integer type
  static EnumType enumType() {
    hid_t tid = H5Tenum_create( AtomicType::atomicType<T>().id() ) ;
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

#endif // HDF5PP_TYPE_H
