#ifndef HDF5PP_GROUPITER_H
#define HDF5PP_GROUPITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GroupIter.
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
#include "hdf5pp/Group.h"

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
 *  @brief Class which implements iteration over groups in HDF5 group/file. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Group
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class GroupIter  {
public:

  /// Enum specifying which links to include
  enum LinkType {
    HardLink = 0x1,
    SoftLink = 0x2,
    Any = 0x3
  };

  /**
   *  @brief Constructor from existing group object.
   *  
   *  @param[in] group     Group object which is iterated
   *  @param[in] type      Type of links to return
   *  
   *  Iterator will return all direct sub-groups of the specified group.
   */
  GroupIter (const Group& group, LinkType type = Any);

  // Destructor
  ~GroupIter () ;

  /**
   *  @brief Returns next group
   *  
   *  If there are no more groups then it returns invalid group object.
   */
  Group next();
  
protected:

private:

  Group m_group;        ///< Group object
  LinkType m_type;      ///< type of links to include in iteration
  hsize_t m_nlinks;     ///< Total number of links in a group
  hsize_t m_idx;        ///< Current index
};

} // namespace hdf5pp

#endif // HDF5PP_GROUPITER_H
