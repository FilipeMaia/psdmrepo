//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Issue...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ErrSvc/Issue.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ErrSvc {

//----------------
// Constructors --
//----------------
Issue::Issue (const Context& ctx, const std::string& message)
  : std::exception()
  , m_ctx(ctx)
  , m_message(message)
  , m_fullMessage()
{
  std::ostringstream str ;
  str << message << " [" << ctx << "]";
  m_fullMessage = str.str();
}

//--------------
// Destructor --
//--------------
Issue::~Issue () throw()
{
}

// Implements std::exception::what()
const char* 
Issue::what() const throw()
{
  return m_fullMessage.c_str();
}

// Stream insertion operator
std::ostream&
operator<<(std::ostream& out, const Issue& issue)
{
  return out << issue.what();
}

} // namespace ErrSvc
