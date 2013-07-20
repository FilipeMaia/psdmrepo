//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptBase
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

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptBase.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

// Destructor
AppCmdOptBase::~AppCmdOptBase( )
{
}

// Define an option.
AppCmdOptBase::AppCmdOptBase(const std::string& optNames,
    const std::string& name,
    const std::string& descr)
  : _shortOpt('\0')
  , _longOpt()
  , _name(name)
  , _descr(descr)
{
  std::string::size_type p0 = 0;
  std::string::size_type p1 = optNames.find(',');
  while (p0 != std::string::npos) {
    std::string opt;
    if (p1 == std::string::npos) {
      opt = optNames.substr(p0);
      p0 = p1;
    } else {
      opt = optNames.substr(p0, p1-p0);
      p0 = p1+1;
      p1 = optNames.find(',', p0);
    }

    // check option format
    if (not opt.empty() and opt[0] == '-') throw AppCmdOptNameException(opt);

    if (opt.size() == 1) {
      _shortOpt = opt[0];
    } else if (opt.size() > 1) {
      _longOpt = opt;
    }
  }
}

} // namespace AppUtils
