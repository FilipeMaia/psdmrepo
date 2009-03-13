#ifndef APPUTILS_APPCMDEXCEPTIONS_H
#define APPUTILS_APPCMDEXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdExceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdexcept>

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
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: template!C++!h 4 2008-10-08 19:27:36Z salnikov $
 *
 *  @author Andrei Salnikov
 */

namespace AppUtils {

class AppCmdException : public std::runtime_error {
public:
  AppCmdException ( const std::string& msg ) ;
};

class AppCmdTypeCvtException : public AppCmdException {
public:
  AppCmdTypeCvtException ( const std::string& string, const std::string& type ) ;
};

class AppCmdOptReservedException : public AppCmdException {
public:
  AppCmdOptReservedException ( char option ) ;
  AppCmdOptReservedException ( const std::string& option ) ;
};

class AppCmdOptDefinedException : public AppCmdException {
public:
  AppCmdOptDefinedException ( char option ) ;
  AppCmdOptDefinedException ( const std::string& option ) ;
};

class AppCmdOptUnknownException : public AppCmdException {
public:
  AppCmdOptUnknownException ( char option ) ;
  AppCmdOptUnknownException ( const std::string& option ) ;
};

class AppCmdArgOrderException : public AppCmdException {
public:
  AppCmdArgOrderException ( const std::string& arg ) ;
};

} // namespace AppUtils

#endif // APPUTILS_APPCMDEXCEPTIONS_H
