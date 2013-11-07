#ifndef PSEVT_TYPEINFOUTILS_H
#define PSEVT_TYPEINFOUTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: 
//
// Description:
//	Class TypeInfoUtils
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <typeinfo>

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

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Class with utility functions to manipulate C++ type_info.
 *
 *  @see EventKey
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

namespace TypeInfoUtils {

// less for type_info *, can use for std::map  
class lessTypeInfoPtr { 
 public: 
  bool operator()(const std::type_info *a, const std::type_info *b) { 
    return a->before( *b); 
  } 
}; 

std::string typeInfoRealName(const std::type_info *);

} // namespace TypeInfoUtil

} // namespace PSEvt


#endif // PSEVT_EVENTKEY_H
