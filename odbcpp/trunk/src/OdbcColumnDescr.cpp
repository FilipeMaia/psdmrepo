//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcColumnDescr...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcColumnDescr.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcException.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace odbcpp {

//----------------
// Constructors --
//----------------
OdbcColumnDescr::OdbcColumnDescr ( OdbcHandle<OdbcStmt> stmtH, SQLSMALLINT colNum )
  : m_colNum(colNum)
{
  SQLCHAR nameBuf[128] ;
  SQLSMALLINT nameLen ;
  SQLRETURN r = SQLDescribeCol ( *stmtH,
                                  m_colNum,
                                  nameBuf,
                                  sizeof nameBuf,
                                  &nameLen,
                                  &m_dataType,
                                  &m_columnSize,
                                  &m_decimalDigits,
                                  &m_nullable );
  OdbcStatusCheck ( r, stmtH ) ;

  m_name = std::string( (char*)nameBuf, nameLen ) ;
}

//--------------
// Destructor --
//--------------
OdbcColumnDescr::~OdbcColumnDescr ()
{
}

} // namespace odbcpp
