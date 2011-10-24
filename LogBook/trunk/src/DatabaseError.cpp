//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
//
// Description:
//	Class DatabaseError...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "LogBook/DatabaseError.h"

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

DatabaseError::DatabaseError (const std::string& reason) :
    std::exception(),
    m_reason("LogBook::DatabaseError: " + reason)
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

} // namespace LogBook
