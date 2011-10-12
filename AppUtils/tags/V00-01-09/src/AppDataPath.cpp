//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppDataPath...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppDataPath.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

//----------------
// Constructors --
//----------------
AppDataPath::AppDataPath(const std::string& relPath)
  : m_path()
{
  // get $SIT_DATA
  const char* dataPath = getenv("SIT_DATA");
  if (not dataPath) return;

  // split SIT_DATA path on :
  std::list<std::string> paths;
  boost::split(paths, dataPath, boost::is_any_of(":"));

  // find first existing file
  for (std::list<std::string>::const_iterator it = paths.begin(); it != paths.end(); ++ it) {

    fs::path path = *it;
    path /= relPath;

    if (fs::exists(path)) {
      m_path = path.string();
      break;
    }

  }
}

} // namespace AppUtils
