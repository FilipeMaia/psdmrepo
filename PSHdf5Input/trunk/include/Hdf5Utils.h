#ifndef PSHDF5INPUT_HDF5UTILS_H
#define PSHDF5INPUT_HDF5UTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5Utils.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "PSTime/Time.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/// @addtogroup PSHdf5Input

/**
 *  @ingroup PSHdf5Input
 *
 *  @brief Utility class with several helper methods for dealing with HDF5 data.
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5Utils  {
public:

  /**
   *  @brief Get attribute value from HDF5 group
   */
  template <typename T>
  static T getAttr(hdf5pp::Group& grp, const std::string& attr, T def = T())
  {
    if (grp.hasAttr(attr)) return grp.openAttr<T>(attr).read();
    return def;
  }

  /**
   *  @brief get value of time attribute.
   *
   *  Time attribute is specified with two integer attributes: {time}.seconds and
   *  {time}.nanoseconds where {time} string tells what time is it (usually start
   *  or end).
   */
  static PSTime::Time getTime(hdf5pp::Group& grp, const std::string& time);

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5UTILS_H
