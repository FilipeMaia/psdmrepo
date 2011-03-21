//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFileName...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcFileName.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stdlib.h>
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char sep = '-' ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

XtcFileName::XtcFileName()
  : m_path()
  , m_expNum(0)
  , m_run(0)
  , m_stream(0)
  , m_chunk(0)
{
}

XtcFileName::XtcFileName ( const std::string& path )
  : m_path(path)
  , m_expNum(0)
  , m_run(0)
  , m_stream(0)
  , m_chunk(0)
{

  std::string name = basename() ;

  // remove extension
  std::string::size_type n = name.find('.') ;
  if ( n != std::string::npos ) name.erase ( n ) ;

  // file name expected to be eNNN_rNNN_sNNN_chNNN.xtc

  // find underscore before chunk part
  n = name.rfind(sep) ;
  if ( n == std::string::npos || (name.size()-n) < 3 || name[n+1] != 'c' ) return ;
  bool stat ;
  unsigned chunk = _cvt ( name.c_str()+n+2, stat ) ;
  if ( not stat ) return ;

  // remove chunk part
  name.erase(n) ;

  // find underscore before stream part
  n = name.rfind(sep) ;
  if ( n == std::string::npos || (name.size()-n) < 3 || name[n+1] != 's' )  return ;
  unsigned stream = _cvt ( name.c_str()+n+2, stat ) ;
  if ( not stat ) return ;

  // remove stream part
  name.erase(n) ;

  // find underscore before run part
  n = name.rfind(sep) ;
  if ( n == std::string::npos || (name.size()-n) < 3 || name[n+1] != 'r' ) return ;
  unsigned run = _cvt ( name.c_str()+n+2, stat ) ;
  if ( not stat ) return ;

  // remove run part
  name.erase(n) ;

  // remaining must be experiment part
  if ( name.size() < 2 || name[0] != 'e' ) return ;
  unsigned expNum = _cvt ( name.c_str()+1, stat ) ;
  if ( not stat ) return ;

  m_expNum = expNum ;
  m_run = run ;
  m_stream = stream ;
  m_chunk = chunk ;
}


// get base name
std::string
XtcFileName::basename() const
{
  std::string name = m_path ;

  // remove dirname
  std::string::size_type n = name.rfind('/') ;
  if ( n != std::string::npos ) name.erase ( 0, n+1 ) ;

  return name ;
}

// compare two names
bool
XtcFileName::operator<( const XtcFileName& other ) const
{
  if ( m_expNum < other.m_expNum ) return true ;
  if ( other.m_expNum < m_expNum ) return false ;

  if ( m_run < other.m_run ) return true ;
  if ( other.m_run < m_run ) return false ;

  if ( m_stream < other.m_stream ) return true ;
  if ( other.m_stream < m_stream ) return false ;

  if ( m_chunk < other.m_chunk ) return true ;
  return false ;

}

unsigned
XtcFileName::_cvt ( const char* ptr, bool& stat ) const
{
  char *eptr = 0 ;
  int val = strtol ( ptr, &eptr, 10 ) ;
  stat = ( *eptr == 0 ) and val >= 0 ;
  return unsigned(val) ;
}

std::ostream&
operator<<(std::ostream& out, const XtcFileName& fn)
{
  return out << fn.path() ;
}

} // namespace XtcInput
