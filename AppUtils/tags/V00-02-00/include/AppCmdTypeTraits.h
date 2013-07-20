//--------------------------------------------------------------------------
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------

#ifndef APPUTILS_APPCMDTYPETRAITS_HH
#define APPUTILS_APPCMDTYPETRAITS_HH

//-------------
// C Headers --
//-------------
#include <limits.h>
#include <float.h>

//---------------
// C++ Headers --
//---------------
#include <cstdlib>
#include <cerrno>
#include <cmath>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"

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
 *  @brief Type traits used by AppCmdLine classes.
 *
 *  Type traits for the command line options and arguments. Every type
 *  to be used as the template parameter for classes AppCmdArg<T> or
 *  AppCmdOpt<T> should provide specialization for the AppCmdTypeTraits<T>
 *  struct.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdArg
 *  @see AppCmdOpt
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

template<class T>
struct AppCmdTypeTraits {
};

/**
 *  Specialization for type long int
 */
template<>
struct AppCmdTypeTraits<long> {
  static long fromString ( const std::string& str ) {
    const char* nptr = str.c_str() ;
    char* end ;
    errno = 0 ; // not thread-safe
    long val = std::strtol ( nptr, &end, 0 ) ;
    // conversion must consume all characters, otherwise it's not successful
    // check errno also and value for overflow/underflow
    if (end== nptr || *end != '\0' ||
        ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN))) || (errno != 0 && val == 0)) {
      throw AppCmdTypeCvtException ( str, "long" ) ;
    }
    return val ;
  }
};

/**
 *  Specialization for type int
 */
template<>
struct AppCmdTypeTraits<int> {
  static int fromString ( const std::string& str ) {
    try {
      long res = AppCmdTypeTraits<long>::fromString( str ) ;
      if ( res >= INT_MIN && res <= INT_MAX ) return int(res) ;
    } catch (AppCmdTypeCvtException& e) {
    }
    throw AppCmdTypeCvtException ( str, "int" ) ;
  }
};

/**
 *  Specialization for type unsigned long
 */
template<>
struct AppCmdTypeTraits<unsigned long> {
  static unsigned long fromString ( const std::string& str ) {
    const char* nptr = str.c_str() ;
    char* end ;
    errno = 0 ;
    unsigned long val = std::strtoul ( nptr, &end, 0 ) ;
    // conversion must consume all characters, otherwise it's not successful
    // check errno also and value for overflow/underflow
    if (end==nptr || *end != '\0' ||
        ((errno==ERANGE && val==ULONG_MAX) || (errno!=0 && val==0))) {
      throw AppCmdTypeCvtException ( str, "unsigned long" ) ;
    }
    return val ;
  }
};

/**
 *  Specialization for type unsigned int
 */
template<>
struct AppCmdTypeTraits<unsigned int> {
  static unsigned int fromString ( const std::string& str ) {
    try {
      unsigned long res = AppCmdTypeTraits<unsigned long>::fromString( str ) ;
      // check the range
      if ( res <= UINT_MAX ) return (unsigned int)( res ) ;
    } catch (AppCmdTypeCvtException& e) {
    }
    throw AppCmdTypeCvtException ( str, "unsigned int" ) ;
  }
};

/**
 *  Specialization for type std::string
 */
template<>
struct AppCmdTypeTraits<std::string> {
  static std::string fromString ( const std::string& str ) {
    return str ;
  }
};

/**
 *  Specialization for type bool
 */
template<>
struct AppCmdTypeTraits<bool> {
  static bool fromString ( const std::string& str ) {
    if ( str == "true" || str == "TRUE" || str == "1" || str == "yes" || str == "YES" ) {
      return true ;
    } else if ( str == "false" || str == "FALSE" || str == "0" || str == "no" || str == "NO" ) {
      return false ;
    } else {
      throw AppCmdTypeCvtException ( str, "bool" ) ;
    }
  }
};

/**
 *  Specialization for type double
 */
template<>
struct AppCmdTypeTraits<double> {
  static double fromString ( const std::string& str ) {
    const char* nptr = str.c_str() ;
    char* end ;
    errno = 0;
    double val = std::strtod ( nptr, &end ) ;
    // conversion must consume all characters, otherwise it's not successful
    // check errno also and value for overflow/underflow
    if (end==nptr || *end != '\0' ||
        ((errno == ERANGE && (val == HUGE_VAL || val == -HUGE_VAL))) || (errno != 0 && val == 0)) {
      throw AppCmdTypeCvtException ( str, "double" ) ;
    }
    return val ;
  }
};

/**
 *  Specialization for type float
 */
template<>
struct AppCmdTypeTraits<float> {
  static float fromString ( const std::string& str ) {
    const char* nptr = str.c_str() ;
    char* end ;
    errno = 0;
    float val = std::strtof ( nptr, &end ) ;
    // conversion must consume all characters, otherwise it's not successful
    // check errno also and value for overflow/underflow
    if (end==nptr || *end != '\0' ||
        ((errno == ERANGE && (val == HUGE_VALF || val == -HUGE_VALF))) || (errno != 0 && val == 0)) {
      throw AppCmdTypeCvtException ( str, "float" ) ;
    }
    return val ;
  }
};

} // namespace AppUtils


#endif  // APPUTILS_APPCMDTYPETRAITS_HH
