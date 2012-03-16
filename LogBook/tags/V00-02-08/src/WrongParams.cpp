//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
//
// Description:
//	Class WrongParams...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "LogBook/WrongParams.h"

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

namespace LogBook {

//----------------
// Constructors --
//----------------

WrongParams::WrongParams (const std::string& reason) :
    std::exception(),
    m_reason ("LogBook::WrongParams: "+reason)
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

} // namespace LogBook
