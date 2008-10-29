//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AppCmdLineTest.cc,v 1.7 2006/02/10 17:55:30 salnikov Exp $
//
// Description:
//	Test suite case for the AppCmdLine & friends.
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	Andy Salnikov		originator
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <string>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdLine.h"
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdArgList.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdOptToggle.h"
#include "AppUtils/AppCmdOptNamedValue.h"
using std::cout;


using namespace AppUtils ;


int main( int argc, char* argv[] )
{
  // Install cmd line parser, cannot fail
  AppCmdLine cmdline ( "command" ) ;

  bool ok ;

  // create a bunch of arguments and add them
  AppCmdArg<std::string> argString1( "name", "specifies the name" ) ;
  ok = cmdline.addArgument ( argString1 ) ;
  assert ( ok ) ;

  AppCmdArg<int> argInt1( "number", "specifies the number" ) ;
  ok = cmdline.addArgument ( argInt1 ) ;
  assert ( ok ) ;

  AppCmdArg<int> argInt2( "number", "specifies the number 2", 1000 ) ;
  ok = cmdline.addArgument ( argInt2 ) ;
  assert ( ok ) ;

  // make a command line
  std::list<std::string> args ;
  args.push_back ( "MyName" ) ;
  args.push_back ( "12345" ) ;

  // first try, stringfor optional argument is mising
  ok = cmdline.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( argString1.value() == "MyName" ) ;
  assert ( argInt1.value() == 12345 ) ;
  assert ( argInt1.valueChanged() == true ) ;
  assert ( argInt2.value() == 1000 ) ;
  assert ( argInt2.valueChanged() == false ) ;


  // add data for optional argument
  args.push_back ( "123" ) ;

  ok = cmdline.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( argString1.value() == "MyName" ) ;
  assert ( argInt1.value() == 12345 ) ;
  assert ( argInt2.value() == 123 ) ;
  assert ( argInt2.valueChanged() == true ) ;

  // one more argument should fail
  args.push_back ( "fail" ) ;

  ok = cmdline.parse ( args.begin(), args.end() ) ;
  assert ( ! ok ) ;

  // add more options
  AppCmdOptIncr optVerbose ( 'v', "verbose", "more noise", 0 ) ;
  ok = cmdline.addOption ( optVerbose ) ;
  assert ( ok ) ;

  AppCmdOptToggle optToggle ( 't', "toggle", "toggle something", false ) ;
  ok = cmdline.addOption ( optToggle ) ;
  assert ( ok ) ;

  AppCmdOpt<int> optInt1 ( 'i', "int", "number", "some number", 123 ) ;
  ok = cmdline.addOption ( optInt1 ) ;
  assert ( ok ) ;

  AppCmdOpt<int> optInt2 ( 'I', "INT", "NUMBER", "some number", 123 ) ;
  ok = cmdline.addOption ( optInt2 ) ;
  assert ( ok ) ;

  AppCmdOpt<std::string> optString1 ( 's', "string", "astring", "some string", "<none>" ) ;
  ok = cmdline.addOption ( optString1 ) ;
  assert ( ok ) ;

  AppCmdOpt<std::string> optString2 ( 'S', "STRING", "Astring", "some string", "<none>" ) ;
  ok = cmdline.addOption ( optString2 ) ;
  assert ( ok ) ;
  ok = cmdline.addOption ( optString2 ) ;
  assert ( ! ok ) ;

  AppCmdOpt<std::string> optString3 ( 'd', "dummy", "Astring", "some string", "<none>" ) ;
  ok = cmdline.addOption ( optString3 ) ;
  assert ( ok ) ;

  // new command line
  args.clear() ;
  args.push_back ( "-v" ) ;
  args.push_back ( "--verbose" ) ;
  args.push_back ( "-t" ) ;
  args.push_back ( "--toggle" ) ;
  args.push_back ( "-vvvt" ) ;
  args.push_back ( "--int=654" ) ;
  args.push_back ( "--INT" ) ;
  args.push_back ( "654" ) ;
  args.push_back ( "-sNone" ) ;
  args.push_back ( "-S" ) ;
  args.push_back ( "NONE" ) ;
  args.push_back ( "--" ) ;
  args.push_back ( "MyName" ) ;
  args.push_back ( "12345" ) ;

  // first try, stringfor optional argument is mising
  ok = cmdline.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( optVerbose.value() == 5 ) ;
  assert ( optToggle.value() == true ) ;
  assert ( optInt1.value() == 654 ) ;
  assert ( optInt2.value() == 654 ) ;
  assert ( optString1.value() == "None" ) ;
  assert ( optString2.value() == "NONE" ) ;
  assert ( optString2.valueChanged() == true ) ;
  assert ( optString3.valueChanged() == false ) ;

  // print usage info about command
  cmdline.usage ( cout ) ;


  // Install one more cmd line parser, cannot fail
  AppCmdLine cmdline1( "command1" ) ;

  // create a bunch of arguments and add them
  AppCmdArgList<std::string> argStringL( "names", "specifies the name(s)" ) ;
  ok = cmdline1.addArgument ( argStringL ) ;
  assert ( ok ) ;

  AppCmdOptList<std::string> optStringL( 'n', "name", "string", "specifies the name(s)" ) ;
  ok = cmdline1.addOption ( optStringL ) ;
  assert ( ok ) ;

  args.clear() ;
  args.push_back ( "-nname1" ) ;
  args.push_back ( "-n" ) ;
  args.push_back ( "name2" ) ;
  args.push_back ( "--name=name3" ) ;
  args.push_back ( "--name" ) ;
  args.push_back ( "name4" ) ;
  args.push_back ( "--name" ) ;
  args.push_back ( "name5,name6,name7,name8" ) ;
  args.push_back ( "name1" ) ;
  args.push_back ( "name2" ) ;
  args.push_back ( "name3" ) ;

  ok = cmdline1.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;

  AppCmdOptList<std::string>::const_iterator obegin = optStringL.begin() ;
  assert ( *obegin == "name1" ) ;
  ++ obegin ;
  assert ( *obegin == "name2" ) ;
  ++ obegin ;
  assert ( *obegin == "name3" ) ;
  ++ obegin ;
  assert ( *obegin == "name4" ) ;
  ++ obegin ;
  assert ( *obegin == "name5" ) ;
  ++ obegin ;
  assert ( *obegin == "name6" ) ;
  ++ obegin ;
  assert ( *obegin == "name7" ) ;
  ++ obegin ;
  assert ( *obegin == "name8" ) ;
  ++ obegin ;
  assert ( obegin == optStringL.end() ) ;

  AppCmdArgList<std::string>::const_iterator begin = argStringL.begin() ;
  assert ( *begin == "name1" ) ;
  ++ begin ;
  assert ( *begin == "name2" ) ;
  ++ begin ;
  assert ( *begin == "name3" ) ;
  ++ begin ;
  assert ( begin == argStringL.end() ) ;

  // print usage info about command
  cmdline1.usage ( cout ) ;

  // should not be able to add -h or --help
  AppCmdOptIncr optHelp ( 'h', "help", "gimme help", 0 ) ;
  ok = cmdline.addOption ( optHelp ) ;
  assert ( ! ok ) ;

  // check how help options work
  args.clear() ;
  args.push_back ( "--help" ) ;
  ok = cmdline1.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( cmdline1.helpWanted() ) ;

  args.clear() ;
  args.push_back ( "-?" ) ;
  ok = cmdline1.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( cmdline1.helpWanted() ) ;

  args.clear() ;
  args.push_back ( "-h" ) ;
  ok = cmdline1.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( cmdline1.helpWanted() ) ;


  // Install one more cmd line parser, cannot fail
  AppCmdLine cmdline2( "command2" ) ;

  AppCmdOpt<int> optInt21 ( 'i', "int", "number", "some number", 0 ) ;
  ok = cmdline2.addOption ( optInt21 ) ;
  assert ( ok ) ;

  args.clear() ;
  args.push_back ( "-i1" ) ;
  ok = cmdline2.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( optInt21.value() == 1 ) ;

  args.clear() ;
  args.push_back ( "-i-1" ) ;
  ok = cmdline2.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( optInt21.value() == -1 ) ;

  args.clear() ;
  args.push_back ( "-i-i" ) ;
  ok = cmdline2.parse ( args.begin(), args.end() ) ;
  assert ( ! ok ) ;

  // Install one more cmd line parser, cannot fail
  AppCmdLine cmdline3( "command3" ) ;

  AppCmdOptNamedValue<int> optInt31 ( 'o', "option", "string", "one of the zero, one, two", 0 ) ;
  optInt31.add ( "zero", 0 ) ;
  optInt31.add ( "one", 1 ) ;
  optInt31.add ( "two", 2 ) ;
  ok = cmdline3.addOption ( optInt31 ) ;
  assert ( ok ) ;


  args.clear() ;
  args.push_back ( "-ozero" ) ;
  ok = cmdline3.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( optInt31.value() == 0 ) ;

  args.clear() ;
  args.push_back ( "--option=one" ) ;
  ok = cmdline3.parse ( args.begin(), args.end() ) ;
  assert ( ok ) ;
  assert ( optInt31.value() == 1 ) ;

  args.clear() ;
  args.push_back ( "-othree" ) ;
  ok = cmdline3.parse ( args.begin(), args.end() ) ;
  assert ( ! ok ) ;

}
