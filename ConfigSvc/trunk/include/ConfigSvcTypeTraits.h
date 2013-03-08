#ifndef CONFIGSVC_CONFIGSVCTYPETRAITS_H
#define CONFIGSVC_CONFIGSVCTYPETRAITS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvcTypeTraits.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigSvc/Exceptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ConfigSvc {

/// @addtogroup ConfigSvc

/**
 *  @ingroup ConfigSvc
 *
 *  Type traits package for type conversion in ConfigSvc service.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

// non-specialized trait class uses generic conversion algorithms 
template <typename T>
struct ConfigSvcTypeTraits {
  
  static T fromString(const std::string& str, const std::string& sect, const std::string& parm) {
    try {
      return boost::lexical_cast<T>(str);
    } catch (const boost::bad_lexical_cast& ex) {
      throw ExceptionCvtFail(sect, parm, str);
    }
  }
  
};

// specialized trait class for bool 
template <>
struct ConfigSvcTypeTraits<bool> {
  
  static bool fromString(const std::string& str, const std::string& sect, const std::string& parm) {
    std::string lstr = boost::algorithm::to_lower_copy(str);
    if (lstr == "yes") return true;
    if (lstr == "no") return false;
    if (lstr == "true") return true;
    if (lstr == "false") return false;
    if (lstr == "on") return true;
    if (lstr == "off") return false;
    try {
      return boost::lexical_cast<bool>(str);
    } catch (const boost::bad_lexical_cast& ex) {
      throw ExceptionCvtFail(sect, parm, str);
    }
  }
  
};

// specialized trait class for strings
template <>
struct ConfigSvcTypeTraits<std::string> {
  
  static const std::string& fromString(const std::string& str, const std::string& sect, const std::string& parm) {
    return str;
  }
  
};

// specialized trait class for strings
template <>
struct ConfigSvcTypeTraits<const char*> {
  // this assumes that lifetime of the string is longer than of the returned pointer
  static const char* fromString(const std::string& str, const std::string& sect, const std::string& parm) {
    return str.c_str();
  }  
};


} // namespace ConfigSvc

#endif // CONFIGSVC_CONFIGSVCTYPETRAITS_H
