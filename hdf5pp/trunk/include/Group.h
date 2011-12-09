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
#include "hdf5pp/DataSet.h"
#include "hdf5pp/DataSpace.h"
#include "hdf5pp/PListDataSetAccess.h"
#include "hdf5pp/PListDataSetCreate.h"

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

  // Default constructor
  Group() {}

  // Destructor
  ~Group () ;

  /// Create new group, group name treated as relative to this group
  /// (if not absolute).
  Group createGroup ( const std::string& name ) {
    return createGroup ( *m_id, name ) ;
  }

  /// Open existing group, group name treated as relative to this group
  /// (if not absolute).
  Group openGroup ( const std::string& name ) {
    return openGroup ( *m_id, name ) ;
  }

  /// Determines if the group has a child (link) with the given nae
  bool hasChild ( const std::string& name ) const ;

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

  /// check if attribute exists
  bool hasAttr ( const std::string& name ) {
    return H5Aexists(*m_id, name.c_str()) > 0;
  }

  // create new data set, type is determined by the template type
  template <typename T>
  DataSet<T> createDataSet ( const std::string& name,
                             const DataSpace& dspc,
                             const PListDataSetCreate& plistDScreate = PListDataSetCreate(),
                             const PListDataSetAccess& plistDSaccess = PListDataSetAccess())
  {
    return DataSet<T>::createDataSet ( *m_id, name, TypeTraits<T>::stored_type(), dspc, plistDScreate, plistDSaccess ) ;
  }

  // create new data set, type is determined by explicit parameter
  template <typename T>
  DataSet<T> createDataSet ( const std::string& name,
                             const Type& type,
                             const DataSpace& dspc,
                             const PListDataSetCreate& plistDScreate = PListDataSetCreate(),
                             const PListDataSetAccess& plistDSaccess = PListDataSetAccess())
  {
    return DataSet<T>::createDataSet ( *m_id, name, type, dspc, plistDScreate, plistDSaccess ) ;
  }

  // open existing data set
  template <typename T>
  DataSet<T> openDataSet ( const std::string& name )
  {
    return DataSet<T>::openDataSet ( *m_id, name ) ;
  }

  // close the group
  void close() ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }
  
  // get group name (absolute, may be ambiguous)
  std::string name() const;

  // get group name (relative to some parent)
  std::string basename() const;

  // groups can be used as keys for associative containers, need compare operators
  bool operator<( const Group& other ) const ;
  bool operator==( const Group& other ) const ;
  bool operator!=( const Group& other ) const { return ! this->operator==(other) ; }

protected:

  // allow these guys to call my factory methods
  friend class File ;
  friend class GroupIter ;
  friend class NameIter ;

  // factory methods
  static Group createGroup ( hid_t parent, const std::string& name ) ;
  static Group openGroup ( hid_t parent, const std::string& name ) ;

  // constructor
  Group ( hid_t grp ) ;

private:

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_GROUP_H
