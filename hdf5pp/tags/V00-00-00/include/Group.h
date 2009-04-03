#ifndef HDF5PP_GROUP_H
#define HDF5PP_GROUP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Group.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/Attribute.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Class representing HDF5 group.
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

class File ;

class Group  {
public:

  // Destructor
  ~Group () ;

  /// Create new group, group name treated as relative to this group
  /// (if not absolute).
  Group newGroup ( const std::string& name ) {
    return createGroup ( *m_id, name ) ;
  }

  /// Open existing group, group name treated as relative to this group
  /// (if not absolute).
  Group openGroup ( const std::string& name ) {
    return openGroup ( *m_id, name ) ;
  }

  /// create attribute for this group
  template <typename T>
  Attribute<T> createAttr ( const std::string& name, const DataSpace& dspc = DataSpace::makeScalar() ) {
    return Attribute<T>::createAttr ( *m_id, name, dspc ) ;
  }

  /// open existing attribute
  template <typename T>
  Attribute<T> openAttr ( const std::string& name ) {
    return Attribute<T>::openAttr ( *m_id, name ) ;
  }

protected:

  // allow this guy to call my factory methods
  friend class File ;

  // factory methods
  static Group createGroup ( hid_t parent, const std::string& name ) ;
  static Group openGroup ( hid_t parent, const std::string& name ) ;

  // constructor
  Group ( hid_t grp ) ;

private:

  // deleter for  boost smart pointer
  struct GroupPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Gclose ( *id );
      delete id ;
    }
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_GROUP_H
