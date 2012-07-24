#ifndef ODBCPP_ODBCCOLUMNDESCR_H
#define ODBCPP_ODBCCOLUMNDESCR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcColumnDescr.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcHandle.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  C++ interface for ODBC column descriptor
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace odbcpp {

class OdbcColumnDescr  {
public:

  // Default constructor
  OdbcColumnDescr ( OdbcHandle<OdbcStmt> stmtH, SQLSMALLINT colNum ) ;

  // Destructor
  ~OdbcColumnDescr () ;

  const std::string& columnName() const { return m_name ; }
  SQLSMALLINT columnNumber() const { return m_colNum ; }
  SQLSMALLINT dataType() const { return m_dataType ; }
  SQLUINTEGER columnSize() const { return m_columnSize ; }
  SQLSMALLINT decimalDigits() const { return m_decimalDigits ; }
  SQLSMALLINT nullable() const { return m_nullable ; }

protected:

private:

  // Data members
  std::string m_name ;
  SQLSMALLINT m_colNum ;
  SQLSMALLINT m_dataType ;
  SQLULEN     m_columnSize ;
  SQLSMALLINT m_decimalDigits ;
  SQLSMALLINT m_nullable ;

};

} // namespace odbcpp

#endif // ODBCPP_ODBCCOLUMNDESCR_H
