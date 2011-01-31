//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Context...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ErrSvc/Context.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

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
Context::Context (const char* file, int line, const char* func)
  : m_file(file)
  , m_func(func)
  , m_line(line)
{
}

// Stream insertion operator
std::ostream&
operator<<(std::ostream& out, const Context& ctx)
{
  return out << "in function " << ctx.function() << " at " 
             << ctx.file() << ":" << ctx.line();
}

} // namespace ErrSvc
