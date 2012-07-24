#ifndef ODBCPP_ODBCEXCEPTION_H
#define ODBCPP_ODBCEXCEPTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcExceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <stdexcept>

//----------------------
// Base Class Headers --
//----------------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcHandle.h"
#include "odbcpp/OdbcLog.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Exception classes for odbcpp package
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace odbcpp {

class OdbcException : public std::runtime_error {
public:

  // Constructor from the string message
  OdbcException ( const std::string& msg, const std::string& context )
    : std::runtime_error( context+": "+msg ) {}

  // constructor from a handle containing error info
  template <typename HType>
  OdbcException ( OdbcHandle<HType> h, const char* context )
    : std::runtime_error( h2msg ( h.get(), OdbcHandle<HType>::typecode, context ) ) {}

  // Destructor
  virtual ~OdbcException () throw() {}

protected:

  static std::string h2msg ( SQLHANDLE* h, int typecode, const char* context ) ;

};

} // namespace odbcpp

#define OdbcExceptionCheckSringify2(str) #str
#define OdbcExceptionCheckSringify(str) OdbcExceptionCheckSringify2(str)
#define OdbcExceptionThrow(handleOrMsg) \
  do { \
    const char* ctx = __FILE__ ":" OdbcExceptionCheckSringify(__LINE__) ; \
    if ( const char* r = strrchr ( ctx, '/' ) ) ctx = r+1 ; \
    throw odbcpp::OdbcException ( handleOrMsg, ctx ) ; \
  } while(false)
#define OdbcStatusCheck(status,handleOrMsg) \
  do { \
    if ( not SQL_SUCCEEDED(status) ) {\
      OdbcExceptionThrow(handleOrMsg); \
    }\
  } while(false)
#define OdbcStatusCheckMsg(status,errH,msg) \
  do { \
    if ( not SQL_SUCCEEDED(status) ) {\
      OdbcLog ( error, msg ); \
      OdbcExceptionThrow(errH); \
    } \
  } while(false)
#define OdbcHandleCheck(handle,parentHandleOrMsg) \
  do { \
    if ( not handle ) { \
      OdbcExceptionThrow(parentHandleOrMsg); \
    } \
  } while(false)
#define OdbcHandleCheckMsg(handle,parentH,msg) \
  do { \
    if ( not handle ) {\
      OdbcLog( error, msg ); \
      OdbcExceptionThrow(parentH); \
    } \
  } while(false)

#endif // ODBCPP_ODBCEXCEPTION_H
