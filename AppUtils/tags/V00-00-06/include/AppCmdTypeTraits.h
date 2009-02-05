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
extern "C" {
#include <limits.h>
#include <float.h>
}

//---------------
// C++ Headers --
//---------------

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


/**
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

namespace AppUtils {

template<class T>
struct AppCmdTypeTraits {
};

/**
 *  Specialization for type long int
 */
template<>
struct AppCmdTypeTraits<long> {
  static std::pair<long,bool> fromString ( const std::string& str ) {
    char* end ;
    long res = strtol ( str.c_str(), &end, 0 ) ;
    // convertion must consume all characters, otherwise it's not sucessfull
    bool success = *end == '\0' ;
    return std::pair<long,bool> ( res, success ) ;
  }
};

/**
 *  Specialization for type int
 */
template<>
struct AppCmdTypeTraits<int> {
  static std::pair<int,bool> fromString ( const std::string& str ) {
    std::pair<long,bool> res = AppCmdTypeTraits<long>::fromString( str ) ;
    if ( res.second ) {
      // check the range
      if ( res.first >= INT_MIN && res.first <= INT_MAX ) {
	return std::pair<int,bool>( res.first, res.second ) ;
      }
    }
    return std::pair<int,bool>( res.first, false ) ;
  }
};

/**
 *  Specialization for type unsigned long
 */
template<>
struct AppCmdTypeTraits<unsigned long> {
  static std::pair<unsigned long,bool> fromString ( const std::string& str ) {
    char* end ;
    unsigned long res = strtoul ( str.c_str(), &end, 0 ) ;
    // convertion must consume all characters, otherwise it's not sucessfull
    bool success = *end == '\0' ;
    return std::pair<unsigned long,bool> ( res, success ) ;
  }
};

/**
 *  Specialization for type unsigned int
 */
template<>
struct AppCmdTypeTraits<unsigned int> {
  static std::pair<unsigned int,bool> fromString ( const std::string& str ) {
    std::pair<unsigned long,bool> res = AppCmdTypeTraits<unsigned long>::fromString( str ) ;
    if ( res.second ) {
      // check the range
      if ( res.first <= UINT_MAX ) {
	return std::pair<unsigned int,bool>( res.first, res.second ) ;
      }
    }
    return std::pair<unsigned int,bool>( res.first, false ) ;
  }
};

/**
 *  Specialization for type std::string
 */
template<>
struct AppCmdTypeTraits<std::string> {
  static std::pair<std::string,bool> fromString ( const std::string& str ) {
    return std::pair<std::string,bool> ( str, true ) ;
  }
};

/**
 *  Specialization for type bool
 */
template<>
struct AppCmdTypeTraits<bool> {
  static std::pair<bool,bool> fromString ( const std::string& str ) {
    if ( str == "true" || str == "TRUE" || str == "1" || str == "yes" || str == "YES" ) {
      return std::pair<bool,bool>( true, true ) ;
    } else if ( str == "false" || str == "FALSE" || str == "0" || str == "no" || str == "NO" ) {
      return std::pair<bool,bool>( false, true ) ;
    } else {
      return std::pair<bool,bool>( false, false ) ;
    }
  }
};

/**
 *  Specialization for type double
 */
template<>
struct AppCmdTypeTraits<double> {
  static std::pair<double,bool> fromString ( const std::string& str ) {
    char* end ;
    double res = strtod ( str.c_str(), &end ) ;
    // convertion must consume all characters, otherwise it's not sucessfull
    bool success = *end == '\0' ;
    return std::pair<double,bool> ( res, success ) ;
  }
};

/**
 *  Specialization for type float
 */
template<>
struct AppCmdTypeTraits<float> {
  static std::pair<float,bool> fromString ( const std::string& str ) {
    std::pair<double,bool> res = AppCmdTypeTraits<double>::fromString( str ) ;
    if ( res.second ) {
      // check the range
      if ( res.first >= -FLT_MAX && res.first <= FLT_MAX ) {
        return std::pair<float,bool>( res.first, res.second ) ;
      }
    }
    return std::pair<float,bool>( res.first, false ) ;
  }
};

} // namespace AppUtils


#endif  // APPUTILS_APPCMDTYPETRAITS_HH
