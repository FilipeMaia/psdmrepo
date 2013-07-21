//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class WrongParams...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "SciMD/WrongParams.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace SciMD {

//----------------
// Constructors --
//----------------

WrongParams::WrongParams (const std::string& reason) :
    std::exception(),
    m_reason ("SciMD::WrongParams: "+reason)
{ }

//--------------
// Destructor --
//--------------

WrongParams::~WrongParams () throw ()
{ }

//-----------
// Methods --
//-----------

const char*
WrongParams::what () const throw ()
{
    return m_reason.c_str() ;
}

} // namespace SciMD
