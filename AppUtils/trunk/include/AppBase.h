#ifndef APPUTILS_APPBASE_H
#define APPUTILS_APPBASE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppBase.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "AppUtils/AppCmdLine.h"
#include "AppUtils/AppCmdOptIncr.h"

//
// Convenience macro for defining main() function which "runs" given app class
//
#define APPUTILS_MAIN(CLASS) \
  int main( int argc, char* argv[] ) \
  {\
    CLASS app ( argv[0] ) ; \
    return app.run ( argc, argv ) ; \
  }


//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: template!C++!h 4 2008-10-08 19:27:36Z salnikov $
 *
 *  @author Andrei Salnikov
 */

namespace AppUtils {

class AppBase  {
public:

  // Default constructor
  explicit AppBase ( const std::string& appName ) ;

  // Destructor
  virtual ~AppBase () ;

  /**
   *  Run the application
   */
  int run ( int argc, char** argv ) ;

protected:

  /**
   *  add command line option or argument, typically called from subclass constructor
   */
  bool addOption ( AppCmdOptBase& option ) ;
  bool addArgument ( AppCmdArgBase& arg ) ;

  /**
   *  Method called before runApp, can be overriden in subclasses.
   *  Usually if you override it, call base class method too.
   */
  virtual int preRunApp () ;

  /**
   *  Will be implemented in subclasses
   */
  virtual int runApp () = 0 ;

  /**
   *  Method called after runApp, can be overriden in subclasses.
   *  Usually if you override it, call base class method too.
   */
  virtual int postRunApp () ;

  /**
   *  print some additional info after the usage information is printed.
   */
  virtual void moreUsage ( std::ostream& out ) const ;

private:

  // Data members
  AppCmdLine _cmdline ;
  AppCmdOptIncr _optVerbose ;
  AppCmdOptIncr _optQuiet ;

  // Copy constructor and assignment are disabled by default
  AppBase ( const AppBase& ) ;
  AppBase operator = ( const AppBase& ) ;

};

} // namespace AppUtils

#endif // APPUTILS_APPBASE_H
