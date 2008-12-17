#ifndef ODBCPP_ODBCTYPETRAITS_H
#define ODBCPP_ODBCTYPETRAITS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcTypeTraits.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "unixodbc/sql.h"
#include "unixodbc/sqlext.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Type library for ODBC
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

template <typename CppType>
struct OdbcCppType  {
  static SQLSMALLINT cppTypeCode() ;
  static SQLSMALLINT sqlTypeCode() ;
};

#define GEN_ODBC_CPP_TYPE(CPP_TYPE,CPP_TYPE_CODE,SQL_TYPE_CODE) \
  template <> struct OdbcCppType<CPP_TYPE> { \
    static SQLSMALLINT cppTypeCode() { return CPP_TYPE_CODE ; } \
    static SQLSMALLINT sqlTypeCode() { return SQL_TYPE_CODE ; } \
  };

GEN_ODBC_CPP_TYPE(std::string,  SQL_C_CHAR,     SQL_VARCHAR  )
GEN_ODBC_CPP_TYPE(SQLSMALLINT,  SQL_C_SSHORT,   SQL_SMALLINT )
GEN_ODBC_CPP_TYPE(SQLUSMALLINT, SQL_C_USHORT,   SQL_SMALLINT)
GEN_ODBC_CPP_TYPE(int,          SQL_C_SLONG,    SQL_INTEGER )
GEN_ODBC_CPP_TYPE(unsigned int, SQL_C_ULONG,    SQL_INTEGER)
GEN_ODBC_CPP_TYPE(long,         SQL_C_SLONG,    SQL_INTEGER )
GEN_ODBC_CPP_TYPE(unsigned long,SQL_C_ULONG,    SQL_INTEGER)
GEN_ODBC_CPP_TYPE(SQLREAL,      SQL_C_FLOAT,    SQL_REAL )
GEN_ODBC_CPP_TYPE(SQLDOUBLE,    SQL_C_DOUBLE,   SQL_DOUBLE )
GEN_ODBC_CPP_TYPE(SQLSCHAR,     SQL_C_STINYINT, SQL_TINYINT )
GEN_ODBC_CPP_TYPE(SQLCHAR,      SQL_C_UTINYINT, SQL_TINYINT)
GEN_ODBC_CPP_TYPE(SQLBIGINT,    SQL_C_SBIGINT,  SQL_BIGINT )
GEN_ODBC_CPP_TYPE(SQLUBIGINT,   SQL_C_UBIGINT,  SQL_BIGINT )
GEN_ODBC_CPP_TYPE(SQL_DATE_STRUCT, SQL_C_TYPE_DATE, SQL_TYPE_DATE )
GEN_ODBC_CPP_TYPE(SQL_TIME_STRUCT, SQL_C_TYPE_DATE, SQL_TYPE_TIME )
GEN_ODBC_CPP_TYPE(SQL_TIMESTAMP_STRUCT, SQL_C_TYPE_DATE, SQL_TYPE_TIMESTAMP )

#undef GEN_ODBC_CPP_TYPE

} // namespace odbcpp

#endif // ODBCPP_ODBCTYPETRAITS_H
