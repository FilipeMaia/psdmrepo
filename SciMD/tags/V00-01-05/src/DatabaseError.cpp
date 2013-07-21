//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DatabaseError...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "SciMD/DatabaseError.h"

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

DatabaseError::DatabaseError (const std::string& reason) :
    std::exception(),
    m_reason("SciMD::DatabaseError: " + reason)
{ }

//--------------
// Destructor --
//--------------

DatabaseError::~DatabaseError () throw ()
{ }

//-----------
// Methods --
//-----------

const char*
DatabaseError::what () const throw ()
{
    return m_reason.c_str() ;
}

} // namespace SciMD
