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
#include <iosfwd>
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

namespace hdf5pp {

class File ;

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  Class representing HDF5 group.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

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

  /// Determines if the group has a child (link) with the given name
  bool hasChild ( const std::string& name ) const ;

  // get parent for this group, returns non-valid object if no parent group exists
  Group parent() const ;

  /// create attribute for this group
  template <typename T>
  Attribute<T> createAttr ( const std::string& name, const DataSpace& dspc = DataSpace::makeScalar() ) {
    return Attribute<T>::createAttr ( *m_id, name, dspc ) ;
  }

  /// open existing attribute, returns non-valid attribute if does not exist
  template <typename T>
  Attribute<T> openAttr ( const std::string& name ) const {
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

  /**
   *   Create soft link
   *
   *   @param[in] targetPath Path to the link target, can be absolute or relative to this group
   *   @param[in] linkName   Name of this link, relative to this group
   */
  void makeSoftLink(const std::string& targetPath, const std::string& linkName);

  // close the group
  void close() ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }
  
  // get group name (absolute, may be ambiguous)
  std::string name() const;

  // get group id
  hid_t id() const { return *m_id; }

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

  // factory methods
  static Group createGroup ( hid_t parent, const std::string& name ) ;
  static Group openGroup ( hid_t parent, const std::string& name ) ;

  // constructor
  Group ( hid_t grp ) ;

private:

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

/// Insertion operator dumps name and ID of the group.
std::ostream&
operator<<(std::ostream& out, const Group& grp);

} // namespace hdf5pp

#endif // HDF5PP_GROUP_H
