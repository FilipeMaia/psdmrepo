// ==================================================
#include <string>
#include <iostream>

#include "AppUtils/AppBase.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptBool.h"
#include "AppUtils/AppCmdArgList.h"
#include "MsgLogger/MsgLogger.h"

class EchoApp : public AppUtils::AppBase {
public:
  EchoApp( const std::string& appname ) ;
  ~EchoApp() {}

protected:

  int runApp() ;

private :
  AppUtils::AppCmdOptBool m_newLine ;
  AppUtils::AppCmdOpt<std::string> m_sep ;
  AppUtils::AppCmdArgList<std::string> m_args ;
};

EchoApp::EchoApp( const std::string& appname )
  : AppUtils::AppBase(appname)
  , m_newLine(parser(), "n", "disable output of new line", true)
  , m_sep(parser(), "separators", "string", "output word separator", " " )
  , m_args(parser(), "strings", "the list of strings to print", std::vector<std::string>())
{
}

int EchoApp::runApp()
{
  MsgLogRoot( debug, "Starting with newLine=" << m_newLine.value() <<
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
  if ( m_newLine.value() ) std::cout << '\n' ;

  return 0 ;
}

// this macro generates main() which runs above application
APPUTILS_MAIN(EchoApp)
// ==================================================
