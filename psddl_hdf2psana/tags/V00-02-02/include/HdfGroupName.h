#ifndef PSDDL_HDF2PSANA_HDFGROUPNAME_H
#define PSDDL_HDF2PSANA_HDFGROUPNAME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfGroupName.
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
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/TypeId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_hdf2psana {

/// @addtogroup psddl_hdf2psana

/**
 *  @ingroup psddl_hdf2psana
 *
 *  @brief Utility class that converts group names to thingslike
 *  source address or TypeId.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class HdfGroupName  {
public:

  // Get TypeId from group name
  static Pds::TypeId nameToTypeId(const std::string& name);

  // Get Src from group name
  static Pds::Src nameToSource(const std::string& name);

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_HDFGROUPNAME_H
