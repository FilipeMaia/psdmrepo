#ifndef HDF5PP_FILE_H
#define HDF5PP_FILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class File.
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
#include "hdf5pp/Group.h"
#include "hdf5pp/PListFileAccess.h"
#include "hdf5pp/PListFileCreate.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Class that represents HDF5 file. Has an interface for the usual
 *  file operation: open, create, make group, open group.
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

class File  {
public:

  enum CreateMode { Truncate, Exclusive } ;
  enum OpenMode { Read, Update } ;

  /**
   *  Create new HDF5 file.
   */
  static File create( const std::string& path,
                        CreateMode mode,
                        const PListFileCreate& plCreate = PListFileCreate(),
                        const PListFileAccess& plAccess = PListFileAccess() ) ;
  /**
   *  open existing HDF5 file.
   */
  static File open( const std::string& path,
                     OpenMode mode,
                     const PListFileAccess& plAccess = PListFileAccess() ) ;

  /// Default constructor
  File() ;

  // Destructor
  ~File () ;

  /// Create new group, group name treated as relative to the file
  /// (i.e. absolute).
  Group createGroup ( const std::string& name ) {
    return Group::createGroup ( *m_id, name ) ;
  }

  /// Open existing group, group name treated as relative to the file
  /// (i.e. absolute).
  Group openGroup ( const std::string& name ) {
    return Group::openGroup ( *m_id, name ) ;
  }

protected:

  // Constructor
  File ( hid_t id ) ;

private:

  // deleter for  boost smart pointer
  struct FilePtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Fclose ( *id );
      delete id ;
    }
  };

  // Data members
  boost::shared_ptr<hid_t> m_id ;

};

} // namespace hdf5pp

#endif // HDF5PP_FILE_H
