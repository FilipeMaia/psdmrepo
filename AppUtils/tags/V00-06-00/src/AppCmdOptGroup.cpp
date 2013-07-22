//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptGroup...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptGroup.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdLine.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

//----------------
// Constructors --
//----------------
AppCmdOptGroup::AppCmdOptGroup(const std::string& groupName)
  : m_name(groupName)
  , m_options()
{

}

AppCmdOptGroup::AppCmdOptGroup(AppCmdLine& parser, const std::string& groupName)
  : m_name(groupName)
  , m_options()
{
  parser.addGroup(*this);
}

//--------------
// Destructor --
//--------------
AppCmdOptGroup::~AppCmdOptGroup()
{
}

void
AppCmdOptGroup::addOption(AppCmdOptBase& option)
{
  m_options.push_back(&option);
}

} // namespace AppUtils
