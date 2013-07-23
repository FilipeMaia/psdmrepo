#ifndef APPUTILS_APPCMDWORDWRAP_H
#define APPUTILS_APPCMDWORDWRAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdWordWrap.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>

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
 *  @brief Utility class for doing word wrapping of the text.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class AppCmdWordWrap  {
public:

  /**
   *  Constructor takes optional argument representing page width
   *  in characters. If argument is non-positive that constructor tries
   *  to determine page width from current terminal characteristics.
   *  If there is no terminal then default 80 character width is assumed.
   */
  explicit AppCmdWordWrap(int pageWidth=-1);

  /**
   *  Get current page width.
   */
  int pageWidth() const { return m_pageWidth; }

  /**
   *  Split input string into the set of lines so that each line is no
   *  longer than specified width. If width is non-positive then the width
   *  determined by constructor is used. Splitting is done on spaces
   *  and tabs, newline characters cause unconditional split.
   */
  std::vector<std::string> wrap(const std::string& text, int pageWidth=-1) const;

protected:

private:

  int m_pageWidth;

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDWORDWRAP_H
