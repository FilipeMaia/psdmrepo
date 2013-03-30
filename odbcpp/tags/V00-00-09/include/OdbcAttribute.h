#ifndef ODBCPP_ODBCATTRIBUTE_H
#define ODBCPP_ODBCATTRIBUTE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcAttribute.
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
#include "unixodbc/sqlext.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Templated class for ODBC attributes.
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

class OdbcEnvironment ;
class OdbcConnection ;
class OdbcStatement ;

template <typename Type, int Attrib, typename EnvType>
class OdbcAttribute  {
public:

  typedef Type attr_type ;

  // Default constructor
  OdbcAttribute ( EnvType*, unsigned int ) {}
  OdbcAttribute ( attr_type value ) : m_value(value) {}

  attr_type value() const { return m_value ; }
  SQLINTEGER attrib() const { return Attrib ; }

  SQLPOINTER getptr() const { return SQLPOINTER(m_value) ; }
  SQLINTEGER size() const { return 0 ; }

  SQLPOINTER setptr() { return &m_value ; }
  SQLINTEGER setsize() { return 0 ; }
  SQLINTEGER* setsizeptr() { return 0 ; }

protected:

private:

  // Data members
  attr_type m_value ;

};

// partial specialization for string class
template <int Attrib, typename EnvType>
class OdbcAttribute<std::string,Attrib,EnvType> {
public:

  typedef std::string attr_type ;

  // Default constructor
  OdbcAttribute ( EnvType*, SQLINTEGER maxsize )
    : m_buf(new SQLCHAR[maxsize]), m_maxsize(maxsize), m_size(0) {}
  OdbcAttribute( const std::string& value )
    : m_buf(0), m_maxsize(0), m_size(0)
  {
    m_size = value.size() ;
    m_maxsize = m_size+1 ;
    m_buf = new SQLCHAR[m_maxsize] ;
    std::copy ( value.data(), value.data()+m_size, m_buf ) ;
    m_buf[m_size] = '\0' ;
  }
  ~OdbcAttribute() { delete [] m_buf ; }

  OdbcAttribute ( const OdbcAttribute& o )
    : m_buf(new SQLCHAR[o.m_maxsize]), m_maxsize(o.m_maxsize), m_size(o.m_size)
  {
    std::copy ( o.m_buf, o.m_buf+m_maxsize, m_buf ) ;
  }
  OdbcAttribute& operator= ( const OdbcAttribute& o )
  {
    if ( this != & o ) {
      m_buf = new SQLCHAR[o.m_maxsize] ;
      m_maxsize = o.m_maxsize ;
      m_size = o.m_size ;
      std::copy ( o.m_buf, o.m_buf+m_maxsize, m_buf ) ;
    }
    return * this ;
  }

  std::string value() const { return std::string((char*)m_buf,m_size) ; }
  SQLINTEGER attrib() const { return Attrib ; }

  SQLPOINTER getptr() const { return m_buf ; }
  SQLINTEGER size() const { return m_size ; }

  SQLPOINTER setptr() { return m_buf ; }
  SQLINTEGER setsize() { return m_maxsize ; }
  SQLINTEGER* setsizeptr() { return &m_size ; }

protected:

private:

  // Data members
  SQLCHAR* m_buf ;
  SQLINTEGER m_maxsize ;
  SQLINTEGER m_size ;

};

// ======================================
// definitions for environment attributes
// ======================================
typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CONNECTION_POOLING,OdbcEnvironment> ODBC_ATTR_CONNECTION_POOLING ;
#define ODBC_CP_OFF ODBC_ATTR_CONNECTION_POOLING(SQL_CP_OFF)
#define ODBC_CP_ONE_PER_DRIVER ODBC_ATTR_CONNECTION_POOLING(SQL_CP_ONE_PER_DRIVER)
#define ODBC_CP_ONE_PER_HENV ODBC_ATTR_CONNECTION_POOLING(SQL_CP_ONE_PER_HENV)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CP_MATCH,OdbcEnvironment> ODBC_ATTR_CP_MATCH ;
#define ODBC_CP_STRICT_MATCH ODBC_ATTR_CP_MATCH(SQL_CP_STRICT_MATCH)
#define ODBC_CP_RELAXED_MATCH ODBC_ATTR_CP_MATCH(SQL_CP_RELAXED_MATCH)

typedef OdbcAttribute<SQLINTEGER,SQL_ATTR_ODBC_VERSION,OdbcEnvironment> ODBC_ATTR_ODBC_VERSION ;
#define ODBC_OV_ODBC3 ODBC_ATTR_ODBC_VERSION(SQL_OV_ODBC3)
#define ODBC_OV_ODBC2 ODBC_ATTR_ODBC_VERSION(SQL_OV_ODBC2)

typedef OdbcAttribute<SQLINTEGER,SQL_ATTR_OUTPUT_NTS,OdbcEnvironment> ODBC_ATTR_OUTPUT_NTS ;
#define ODBC_NTS_TRUE ODBC_ATTR_OUTPUT_NTS(SQL_TRUE)
#define ODBC_NTS_FALSE ODBC_ATTR_OUTPUT_NTS(SQL_FALSE)

// ======================================
// definitions for connection attributes
// ======================================
typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ACCESS_MODE,OdbcConnection> ODBC_ATTR_ACCESS_MODE ;
#define ODBC_MODE_READ_ONLY ODBC_ATTR_ACCESS_MODE(SQL_MODE_READ_ONLY)
#define ODBC_MODE_READ_WRITE  ODBC_ATTR_ACCESS_MODE(SQL_MODE_READ_WRITE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ASYNC_ENABLE,OdbcConnection> ODBC_ATTR_ASYNC_ENABLE ;
#define ODBC_ASYNC_ENABLE_OFF ODBC_ATTR_ASYNC_ENABLE(SQL_ASYNC_ENABLE_OFF)
#define ODBC_ASYNC_ENABLE_ON  ODBC_ATTR_ASYNC_ENABLE(SQL_ASYNC_ENABLE_ON)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_AUTO_IPD,OdbcConnection> ODBC_ATTR_AUTO_IPD ;
#define ODBC_AUTO_IPD_ON ODBC_ATTR_AUTO_IPD(SQL_TRUE)
#define ODBC_AUTO_IPD_OFF ODBC_ATTR_AUTO_IPD(SQL_FALSE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_AUTOCOMMIT,OdbcConnection> ODBC_ATTR_AUTOCOMMIT ;
#define ODBC_AUTOCOMMIT_OFF ODBC_ATTR_AUTOCOMMIT(SQL_AUTOCOMMIT_OFF)
#define ODBC_AUTOCOMMIT_ON ODBC_ATTR_AUTOCOMMIT(SQL_AUTOCOMMIT_ON)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CONNECTION_DEAD,OdbcConnection> ODBC_ATTR_CONNECTION_DEAD ;
#define ODBC_CD_TRUE ODBC_ATTR_CONNECTION_DEAD(SQL_CD_TRUE)
#define ODBC_CD_FALSE ODBC_ATTR_CONNECTION_DEAD(SQL_CD_FALSE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CONNECTION_TIMEOUT,OdbcConnection> ODBC_ATTR_CONNECTION_TIMEOUT ;
// use it as ODBC_ATTR_CONNECTION_TIMEOUT(5)

typedef OdbcAttribute<std::string,SQL_ATTR_CURRENT_CATALOG,OdbcConnection> ODBC_ATTR_CURRENT_CATALOG ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_LOGIN_TIMEOUT,OdbcConnection> ODBC_ATTR_LOGIN_TIMEOUT ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_METADATA_ID,OdbcConnection> ODBC_ATTR_METADATA_ID ;
#define ODBC_METADATA_ID_ON ODBC_ATTR_METADATA_ID(SQL_TRUE)
#define ODBC_METADATA_ID_OFF ODBC_ATTR_METADATA_ID(SQL_FALSE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ODBC_CURSORS,OdbcConnection> ODBC_ATTR_ODBC_CURSORS ;
#define ODBC_CUR_USE_IF_NEEDED ODBC_ATTR_ODBC_CURSORS(SQL_CUR_USE_IF_NEEDED)
#define ODBC_CUR_USE_ODBC ODBC_ATTR_ODBC_CURSORS(SQL_CUR_USE_ODBC)
#define ODBC_CUR_USE_DRIVER ODBC_ATTR_ODBC_CURSORS(SQL_CUR_USE_DRIVER)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_PACKET_SIZE,OdbcConnection> ODBC_ATTR_PACKET_SIZE ;

//typedef OdbcAttribute<HWND,SQL_ATTR_QUIET_MODE,OdbcConnection> ODBC_ATTR_QUIET_MODE ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_TRACE,OdbcConnection> ODBC_ATTR_TRACE ;
#define ODBC_OPT_TRACE_OFF ODBC_ATTR_TRACE(SQL_OPT_TRACE_OFF)
#define ODBC_OPT_TRACE_ON ODBC_ATTR_TRACE(SQL_OPT_TRACE_ON)

typedef OdbcAttribute<std::string,SQL_ATTR_TRACEFILE,OdbcConnection> ODBC_ATTR_TRACEFILE ;

typedef OdbcAttribute<std::string,SQL_ATTR_TRANSLATE_LIB,OdbcConnection> ODBC_ATTR_TRANSLATE_LIB ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_TRANSLATE_OPTION,OdbcConnection> ODBC_ATTR_TRANSLATE_OPTION ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_TXN_ISOLATION,OdbcConnection> ODBC_ATTR_TXN_ISOLATION ;

// ====================================
// definitions for statement attributes
// ====================================
typedef OdbcAttribute<SQLHDESC,SQL_ATTR_APP_PARAM_DESC,OdbcStatement> ODBC_ATTR_APP_PARAM_DESC ;

typedef OdbcAttribute<SQLHDESC,SQL_ATTR_APP_ROW_DESC,OdbcStatement> ODBC_ATTR_APP_ROW_DESC ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ASYNC_ENABLE,OdbcStatement> ODBC_ATTR_ASYNC_ENABLE_ST ;
#define ODBC_ASYNC_ENABLE_ST_OFF ODBC_ATTR_ASYNC_ENABLE_ST(SQL_ASYNC_ENABLE_OFF)
#define ODBC_ASYNC_ENABLE_ST_ON ODBC_ATTR_ASYNC_ENABLE_ST(SQL_ASYNC_ENABLE_ON)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CONCURRENCY,OdbcStatement> ODBC_ATTR_CONCURRENCY ;
#define ODBC_CONCUR_READ_ONLY ODBC_ATTR_CONCURRENCY(SQL_CONCUR_READ_ONLY)
#define ODBC_CONCUR_LOCK ODBC_ATTR_CONCURRENCY(SQL_CONCUR_LOCK)
#define ODBC_CONCUR_ROWVER ODBC_ATTR_CONCURRENCY(SQL_CONCUR_ROWVER)
#define ODBC_CONCUR_VALUES ODBC_ATTR_CONCURRENCY(SQL_CONCUR_VALUES)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CURSOR_SCROLLABLE,OdbcStatement> ODBC_ATTR_CURSOR_SCROLLABLE ;
#define ODBC_NONSCROLLABLE ODBC_ATTR_CURSOR_SCROLLABLE(SQL_NONSCROLLABLE)
#define ODBC_SCROLLABLE ODBC_ATTR_CURSOR_SCROLLABLE(SQL_SCROLLABLE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CURSOR_SENSITIVITY,OdbcStatement> ODBC_ATTR_CURSOR_SENSITIVITY ;
#define ODBC_UNSPECIFIED ODBC_ATTR_CURSOR_SENSITIVITY(SQL_UNSPECIFIED)
#define ODBC_INSENSITIVE ODBC_ATTR_CURSOR_SENSITIVITY(SQL_INSENSITIVE)
#define ODBC_SENSITIVE ODBC_ATTR_CURSOR_SENSITIVITY(SQL_SENSITIVE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_CURSOR_TYPE,OdbcStatement> ODBC_ATTR_CURSOR_TYPE ;
#define ODBC_CURSOR_FORWARD_ONLY ODBC_ATTR_CURSOR_TYPE(SQL_CURSOR_FORWARD_ONLY)
#define ODBC_CURSOR_STATIC ODBC_ATTR_CURSOR_TYPE(SQL_CURSOR_STATIC)
#define ODBC_CURSOR_KEYSET_DRIVEN ODBC_ATTR_CURSOR_TYPE(SQL_CURSOR_KEYSET_DRIVEN)
#define ODBC_CURSOR_DYNAMIC ODBC_ATTR_CURSOR_TYPE(SQL_CURSOR_DYNAMIC)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ENABLE_AUTO_IPD,OdbcStatement> ODBC_ATTR_ENABLE_AUTO_IPD ;
#define ODBC_ENABLE_AUTO_IPD ODBC_ATTR_ENABLE_AUTO_IPD(SQL_TRUE)
#define ODBC_DISABLE_AUTO_IPD ODBC_ATTR_ENABLE_AUTO_IPD(SQL_FALSE)

typedef OdbcAttribute<void*,SQL_ATTR_FETCH_BOOKMARK_PTR,OdbcStatement> ODBC_ATTR_FETCH_BOOKMARK_PTR ;

typedef OdbcAttribute<SQLHDESC,SQL_ATTR_IMP_PARAM_DESC,OdbcStatement> ODBC_ATTR_IMP_PARAM_DESC ;

typedef OdbcAttribute<SQLHDESC,SQL_ATTR_IMP_ROW_DESC,OdbcStatement> ODBC_ATTR_IMP_ROW_DESC ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_KEYSET_SIZE,OdbcStatement> ODBC_ATTR_KEYSET_SIZE ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_MAX_LENGTH,OdbcStatement> ODBC_ATTR_MAX_LENGTH ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_MAX_ROWS,OdbcStatement> ODBC_ATTR_MAX_ROWS ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_METADATA_ID,OdbcStatement> ODBC_ATTR_METADATA_ID_ST ;
#define ODBC_METADATA_ID_ST_ON ODBC_ATTR_METADATA_ID_ST(SQL_TRUE)
#define ODBC_METADATA_ID_ST_OFF ODBC_ATTR_METADATA_ID_ST(SQL_FALSE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_NOSCAN,OdbcStatement> ODBC_ATTR_NOSCAN ;
#define ODBC_NOSCAN_OFF ODBC_ATTR_NOSCAN(SQL_NOSCAN_OFF)
#define ODBC_NOSCAN_ON ODBC_ATTR_NOSCAN(SQL_NOSCAN_ON)

typedef OdbcAttribute<SQLUINTEGER*,SQL_ATTR_PARAM_BIND_OFFSET_PTR,OdbcStatement> ODBC_ATTR_PARAM_BIND_OFFSET_PTR ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_PARAM_BIND_TYPE,OdbcStatement> ODBC_ATTR_PARAM_BIND_TYPE ;
#define ODBC_PARAM_BIND_BY_COLUMN ODBC_ATTR_PARAM_BIND_TYPE(SQL_PARAM_BIND_BY_COLUMN)

typedef OdbcAttribute<SQLUSMALLINT*,SQL_ATTR_PARAM_OPERATION_PTR,OdbcStatement> ODBC_ATTR_PARAM_OPERATION_PTR ;

typedef OdbcAttribute<SQLUSMALLINT*,SQL_ATTR_PARAM_STATUS_PTR,OdbcStatement> ODBC_ATTR_PARAM_STATUS_PTR ;

typedef OdbcAttribute<SQLUSMALLINT*,SQL_ATTR_PARAMS_PROCESSED_PTR,OdbcStatement> ODBC_ATTR_PARAMS_PROCESSED_PTR ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_PARAMSET_SIZE,OdbcStatement> ODBC_ATTR_PARAMSET_SIZE ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_QUERY_TIMEOUT,OdbcStatement> ODBC_ATTR_QUERY_TIMEOUT ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_RETRIEVE_DATA,OdbcStatement> ODBC_ATTR_RETRIEVE_DATA ;
#define ODBC_RD_ON ODBC_ATTR_RETRIEVE_DATA(SQL_RD_ON)
#define ODBC_RD_OFF ODBC_ATTR_RETRIEVE_DATA(SQL_RD_OFF)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ROW_ARRAY_SIZE,OdbcStatement> ODBC_ATTR_ROW_ARRAY_SIZE ;

typedef OdbcAttribute<SQLUINTEGER*,SQL_ATTR_ROW_BIND_OFFSET_PTR,OdbcStatement> ODBC_ATTR_ROW_BIND_OFFSET_PTR ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ROW_BIND_TYPE,OdbcStatement> ODBC_ATTR_ROW_BIND_TYPE ;
#define ODBC_BIND_BY_COLUMN ODBC_ATTR_ROW_BIND_TYPE(SQL_BIND_BY_COLUMN)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_ROW_NUMBER,OdbcStatement> ODBC_ATTR_ROW_NUMBER ;

typedef OdbcAttribute<SQLUSMALLINT*,SQL_ATTR_ROW_OPERATION_PTR,OdbcStatement> ODBC_ATTR_ROW_OPERATION_PTR ;

typedef OdbcAttribute<SQLUSMALLINT*,SQL_ATTR_ROW_STATUS_PTR,OdbcStatement> ODBC_ATTR_ROW_STATUS_PTR ;

typedef OdbcAttribute<SQLUINTEGER*,SQL_ATTR_ROWS_FETCHED_PTR,OdbcStatement> ODBC_ATTR_ROWS_FETCHED_PTR ;

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_SIMULATE_CURSOR,OdbcStatement> ODBC_ATTR_SIMULATE_CURSOR ;
#define ODBC_SC_NON_UNIQUE ODBC_ATTR_SIMULATE_CURSOR(SQL_SC_NON_UNIQUE)
#define ODBC_SC_TRY_UNIQUE ODBC_ATTR_SIMULATE_CURSOR(SQL_SC_TRY_UNIQUE)
#define ODBC_SC_UNIQUE ODBC_ATTR_SIMULATE_CURSOR(SQL_SC_UNIQUE)

typedef OdbcAttribute<SQLUINTEGER,SQL_ATTR_USE_BOOKMARKS,OdbcStatement> ODBC_ATTR_USE_BOOKMARKS ;
#define ODBC_UB_OFF ODBC_ATTR_USE_BOOKMARKS(SQL_UB_OFF)
#define ODBC_UB_ON ODBC_ATTR_USE_BOOKMARKS(SQL_UB_ON)


} // namespace odbcpp


#endif // ODBCPP_ODBCATTRIBUTE_H
