#ifndef APPUTILS_APPDATAPATH_H
#define APPUTILS_APPDATAPATH_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppDataPath.
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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief This class represents a path to a file that can be found in
 *  one of the $SIT_DATA locations.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class AppDataPath  {
public:

  /// Constructor takes relative file path
  AppDataPath(const std::string& relPath);

  /// Returns path of the existing file or empty string
  const std::string& path() const { return m_path; }

protected:

private:

  std::string m_path;  ///< Path to the file or empty string
  
};

} // namespace AppUtils

#endif // APPUTILS_APPDATAPATH_H
