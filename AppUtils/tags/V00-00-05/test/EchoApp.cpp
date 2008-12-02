// ==================================================
#include <string>
#include <iostream>

#include "AppUtils/AppBase.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdArgList.h"
#include "MsgLogger/MsgLogger.h"

class EchoApp : public AppUtils::AppBase {
public:
  EchoApp( const std::string& appname ) ;
  ~EchoApp() {}

protected:

  int runApp() ;

private :
  AppUtils::AppCmdOptIncr m_noEcho ;
  AppUtils::AppCmdOpt<std::string> m_sep ;
  AppUtils::AppCmdArgList<std::string> m_args ;
};

EchoApp::EchoApp( const std::string& appname )
  : AppUtils::AppBase(appname)
  , m_noEcho( 'n', "no-new-line", "disable output of new line", 0 )
  , m_sep( 's', "separator", "string", "output word separator", " " )
  , m_args ( "strings", "the list of strings to print", std::list<std::string>() )
{
  addOption( m_noEcho ) ;
  addOption( m_sep ) ;
  addArgument( m_args ) ;
}

int EchoApp::runApp()
{
  MsgLogRoot( debug, "Starting with noEcho=" << m_noEcho.value() <<
                     " and sep=\"" << m_sep.value() << "\"" );

  bool first = true ;
  typedef AppUtils::AppCmdArgList<std::string>::const_iterator Iter ;
  for ( Iter i = m_args.begin() ; i != m_args.end() ; ++i ) {
    if ( first ) {
      first = false ;
    } else {
      std::cout << m_sep.value() ;
    }
    std::cout << *i ;
  }
  if ( not m_noEcho.value() ) std::cout << '\n' ;

  return 0 ;
}

// this macro generates main() which runs above application
APPUTILS_MAIN(EchoApp)
// ==================================================
