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
  // create unlocked type as a copy of type id
  static Type Copy( hid_t tid ) { return Type(H5Tcopy(tid),true); }

  // Default constructor
  Type() ;

  // Destructor
  ~Type () ;

  /// return type id
  hid_t id() const { return *m_id ; }

  /// return size of the type in bytes
  size_t size() const ;

  /// make unlocked copy of the type
  Type copy() const ;

  /// set type size
  void set_size( size_t size ) ;

  /// set type precision
  void set_precision( size_t precision ) ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }

protected:

  // constructor
  Type ( hid_t id, bool doClose ) ;

private:

  // deleter for  boost smart pointer
  struct TypePtrDeleter {
    TypePtrDeleter( bool doClose ) : m_doClose(doClose) {}
    void operator()( hid_t* id ) {
      if ( id and m_doClose ) H5Tclose ( *id );
      delete id ;
    }
    bool m_doClose ;
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_TYPE_H
