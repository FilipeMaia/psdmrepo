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

template <>
class OdbcParam<std::string> {
public:
  OdbcParam () : m_value(0), m_size(SQL_NULL_DATA) {}
  OdbcParam ( const std::string& value ) {
    m_size = value.size() ;
    m_value = new SQLCHAR[m_size] ;
    std::copy ( value.data(), value.data()+m_size, m_value ) ;
  }
  SQLSMALLINT inputOutputType() const { return SQL_PARAM_INPUT ; }
  SQLSMALLINT valueType() const { return OdbcCppType<std::string>::cppTypeCode() ; }
  SQLSMALLINT parameterType() const { return OdbcCppType<std::string>::sqlTypeCode() ; }
  SQLUINTEGER columnSize() const { return m_size ; }
  SQLSMALLINT decimalDigits() const { return 0 ; }
  SQLPOINTER parameterValuePtr() { return SQLPOINTER(m_value) ; }
  SQLINTEGER bufferLength() const { return m_size ; }
  SQLINTEGER* strLenOrIndPtr() { return &m_size ; }
private:
  SQLCHAR* m_value ;
  SQLINTEGER m_size ;
};

template <typename CppType>
class OdbcParamNumeric  {
public:
  OdbcParamNumeric () : m_value(0), m_size(SQL_NULL_DATA) {}
  OdbcParamNumeric ( CppType value ) : m_value(value), m_size(sizeof value) {}

  SQLSMALLINT inputOutputType() const { return SQL_PARAM_INPUT ; }
  SQLSMALLINT valueType() const { return OdbcCppType<CppType>::cppTypeCode() ; }
  SQLSMALLINT parameterType() const { return OdbcCppType<CppType>::sqlTypeCode() ; }
  SQLUINTEGER columnSize() const { return 0 ; }
  SQLSMALLINT decimalDigits() const { return 0 ; }
  SQLPOINTER parameterValuePtr() { return SQLPOINTER(&m_value) ; }
  SQLINTEGER bufferLength() const { return m_size ; }
  SQLINTEGER* strLenOrIndPtr() { return &m_size ; }
private:
  CppType m_value ;
  SQLINTEGER m_size ;
};

#define GEN_NUM_PARAM(CPP_TYPE) \
  template <> class OdbcParam<CPP_TYPE> : public OdbcParamNumeric<CPP_TYPE> {\
  public:\
    OdbcParam( CPP_TYPE value ) : OdbcParamNumeric<CPP_TYPE>( value ) {} \
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
GEN_NUM_PARAM(SQLBIGINT)
GEN_NUM_PARAM(SQLUBIGINT)

#undef GEN_NUM_PARAM

} // namespace odbcpp

#endif // ODBCPP_ODBCPARAM_H
