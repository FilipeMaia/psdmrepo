#ifndef HDF5PP_NAMEITER_H
#define HDF5PP_NAMEITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NameIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

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

/**
 *  @brief Class which implements iteration over link names in HDF5 group.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class NameIter  {
public:

  /**
   *  @brief Constructor from existing group object.
   *  
   *  Iterator will return names of all links in the specified group.
   */
  NameIter (const Group& group) ;

  // Destructor
  ~NameIter () ;

  /**
   *  @brief Returns next name.
   *  
   *  If there are no more groups then empty string will be returned.
   */
  std::string next();
  
protected:

private:

  Group m_group;        ///< Group object
  hsize_t m_nlinks;     ///< Total number of links in a group
  hsize_t m_idx;        ///< Cuyrrent index
};

} // namespace hdf5pp

#endif // HDF5PP_NAMEITER_H
