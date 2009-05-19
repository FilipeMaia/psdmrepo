#ifndef ODBCPP_ODBCPARAM_H
#define ODBCPP_ODBCPARAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcParam.
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
#include "odbcpp/OdbcTypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Statement parameter class.
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

// generic non-specialized code cannot be used
template <typename CppType>
class OdbcParam  {

  // all these methods have to be implemented in every specialization
  SQLSMALLINT inputOutputType() const ;
  SQLSMALLINT valueType() const ;
  SQLSMALLINT parameterType() const ;
  SQLUINTEGER columnSize() const ;
  SQLSMALLINT decimalDigits() const ;
  SQLPOINTER parameterValuePtr() ;
  SQLINTEGER bufferLength() const ;
  SQLINTEGER* strLenOrIndPtr() ;

  void setData ( CppType ) ;
  void setNull () ;
};

//
// Specialization for string class
//
template <>
class OdbcParam<std::string> {
public:
  OdbcParam () : m_value(0), m_size(SQL_NULL_DATA) {}
  OdbcParam ( const std::string& value, unsigned int maxSize = 0 ) {
    m_size = value.size() ;
    m_maxSize = maxSize ;
    if ( m_maxSize < m_size ) m_maxSize = m_size ;
    if ( m_maxSize < 16 ) m_maxSize = 16 ;
    m_value = new SQLCHAR[m_maxSize] ;
    std::copy ( value.data(), value.data()+m_size, m_value ) ;
  }
  OdbcParam ( const char* value, unsigned int maxSize = 0 ) {
    m_size = strlen(value) ;
    m_maxSize = maxSize ;
    if ( m_maxSize < m_size ) m_maxSize = m_size ;
    if ( m_maxSize < 16 ) m_maxSize = 16 ;
    m_value = new SQLCHAR[m_maxSize] ;
    std::copy ( value, value+m_size, m_value ) ;
  }
  ~OdbcParam () { delete m_value ; }

  SQLSMALLINT inputOutputType() const { return SQL_PARAM_INPUT ; }
  SQLSMALLINT valueType() const { return OdbcCppType<std::string>::cppTypeCode() ; }
  SQLSMALLINT parameterType() const { return OdbcCppType<std::string>::sqlTypeCode() ; }
  SQLUINTEGER columnSize() const { return m_size ; }
  SQLSMALLINT decimalDigits() const { return 0 ; }
  SQLPOINTER parameterValuePtr() { return SQLPOINTER(m_value) ; }
  SQLINTEGER bufferLength() const { return m_size ; }
  SQLINTEGER* strLenOrIndPtr() { return &m_size ; }

  void setData ( const std::string& value ) {
    m_size = value.size() ;
    if ( m_size > m_maxSize ) m_size = m_maxSize ;
    std::copy ( value.data(), value.data()+m_size, m_value ) ;
  }
  void setData ( const char* value ) {
    m_size = strlen(value) ;
    if ( m_size > m_maxSize ) m_size = m_maxSize ;
    std::copy ( value, value+m_size, m_value ) ;
  }
  void setNull () { m_size = SQL_NULL_DATA; }

private:
  SQLCHAR* m_value ;
  SQLINTEGER m_maxSize ;
  SQLINTEGER m_size ;
};

//
// Helper base class for the value-based types
//
template <typename CppType>
class OdbcParamValue  {
public:

  OdbcParamValue () : m_value(0), m_size(SQL_NULL_DATA) {}
  OdbcParamValue ( CppType value ) : m_value(value), m_size(sizeof value) {}

  SQLSMALLINT inputOutputType() const { return SQL_PARAM_INPUT ; }
  SQLSMALLINT valueType() const { return OdbcCppType<CppType>::cppTypeCode() ; }
  SQLSMALLINT parameterType() const { return OdbcCppType<CppType>::sqlTypeCode() ; }
  SQLUINTEGER columnSize() const { return 0 ; }
  SQLSMALLINT decimalDigits() const { return 0 ; }
  SQLPOINTER parameterValuePtr() { return SQLPOINTER(&m_value) ; }
  SQLINTEGER bufferLength() const { return m_size ; }
  SQLINTEGER* strLenOrIndPtr() { return &m_size ; }

  void setData ( CppType val ) { m_value = val ; }
  void setNull () { m_size = SQL_NULL_DATA; }

private:
  CppType m_value ;
  SQLINTEGER m_size ;
};

#define GEN_NUM_PARAM(CPP_TYPE) \
  template <> class OdbcParam<CPP_TYPE> : public OdbcParamValue<CPP_TYPE> {\
  public:\
    OdbcParam( CPP_TYPE value ) : OdbcParamValue<CPP_TYPE>( value ) {} \
  } ;

GEN_NUM_PARAM(short)
GEN_NUM_PARAM(unsigned short)
GEN_NUM_PARAM(int)
GEN_NUM_PARAM(unsigned int)
GEN_NUM_PARAM(long)
GEN_NUM_PARAM(unsigned long)
GEN_NUM_PARAM(float)
GEN_NUM_PARAM(double)
GEN_NUM_PARAM(signed char)
GEN_NUM_PARAM(unsigned char)
GEN_NUM_PARAM(long long)
GEN_NUM_PARAM(long long unsigned)
GEN_NUM_PARAM(SQLBIGINT)
GEN_NUM_PARAM(SQLUBIGINT)
GEN_NUM_PARAM(SQL_DATE_STRUCT)
GEN_NUM_PARAM(SQL_TIME_STRUCT)
GEN_NUM_PARAM(SQL_TIMESTAMP_STRUCT)

#undef GEN_NUM_PARAM

} // namespace odbcpp

#endif // ODBCPP_ODBCPARAM_H
