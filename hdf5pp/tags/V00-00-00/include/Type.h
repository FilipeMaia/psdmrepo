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

  // Destructor
  ~Type () ;

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

} // namespace hdf5pp

#endif // HDF5PP_TYPE_H
