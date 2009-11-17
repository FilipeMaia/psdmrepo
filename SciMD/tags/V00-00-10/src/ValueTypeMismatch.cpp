//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ValueTypeMismatch...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "SciMD/ValueTypeMismatch.h"

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

ValueTypeMismatch::ValueTypeMismatch (const std::string& reason) :
    std::exception(),
    m_reason ("SciMD::ValueTypeMismatch: "+reason)
{ }

//--------------
// Destructor --
//--------------

ValueTypeMismatch::~ValueTypeMismatch () throw ()
{ }

//-----------
// Methods --
//-----------

const char*
ValueTypeMismatch::what () const throw ()
{
    return m_reason.c_str() ;
}

} // namespace SciMD
