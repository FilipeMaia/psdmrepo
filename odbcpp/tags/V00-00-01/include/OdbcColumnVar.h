#ifndef ODBCPP_ODBCCOLUMNVAR_H
#define ODBCPP_ODBCCOLUMNVAR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcColumnVar.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcTypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  C++ interface for the bound column variable
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

// non-specialized thing is not very useful
template <typename VarType>
class OdbcColumnVar  {
public:
  SQLSMALLINT targetType() const ;
  SQLPOINTER targetValuePtr() ;
  SQLINTEGER bufferLength() const ;
  SQLLEN* strLenOrIndPtr() ;
  unsigned int nRows() const ;
  bool isNull(unsigned int idx=0) const ;
  VarType value(unsigned int idx=0) const ;
protected:
  OdbcColumnVar () ;
private:
  OdbcColumnVar ( const OdbcColumnVar& ) ;
  OdbcColumnVar operator = ( const OdbcColumnVar& ) ;
};

// base class with some data common to all classes
class OdbcColumnVarBase {
public:
  OdbcColumnVarBase( unsigned int nRows )
    : m_rows(nRows)
    , m_strLenOrInd(new SQLLEN[nRows])
  {}
  ~OdbcColumnVarBase() { delete [] m_strLenOrInd ; }
  SQLLEN* strLenOrIndPtr() { return m_strLenOrInd ; }
  unsigned int nRows() const { return m_rows ; }
  bool isNull(unsigned int idx=0) const { return m_strLenOrInd[idx] = SQL_NULL_DATA ; }
protected:
  SQLINTEGER m_rows ;
  SQLLEN* m_strLenOrInd ;
};

// specialization for string type
template <>
class OdbcColumnVar<std::string> : public OdbcColumnVarBase {
public:
  OdbcColumnVar( int stringSize, unsigned int nRows = 1 )
    : OdbcColumnVarBase(nRows)
    , m_value(new SQLCHAR[stringSize*nRows])
    , m_size(stringSize)
  {}
  ~OdbcColumnVar() { delete [] m_value ; }

  SQLSMALLINT targetType() const { return OdbcCppType<std::string>::cppTypeCode() ; }
  SQLPOINTER targetValuePtr() { return m_value ; }
  SQLINTEGER bufferLength() const { return m_size ; }
  std::string value(unsigned int idx=0) const {
    return std::string( (char*)(m_value+m_size*idx), m_strLenOrInd[idx] ) ;
  }

private:
  SQLCHAR* m_value ;
  SQLINTEGER m_size ;

  OdbcColumnVar ( const OdbcColumnVar& ) ;
  OdbcColumnVar operator = ( const OdbcColumnVar& ) ;
};


// helper class for numeric types
template <typename NumType>
class OdbcColumnVarNumeric : public OdbcColumnVarBase {
public:
  OdbcColumnVarNumeric( unsigned int nRows )
    : OdbcColumnVarBase(nRows)
    , m_value(new NumType[nRows])
  {}
  ~OdbcColumnVarNumeric() { delete [] m_value ; }

  SQLSMALLINT targetType() const { return OdbcCppType<NumType>::cppTypeCode() ; }
  SQLPOINTER targetValuePtr() { return m_value ; }
  SQLINTEGER bufferLength() const { return sizeof(NumType) ; }
  NumType value(unsigned int idx=0) const { return m_value[idx] ; }

private:
  NumType* m_value ;

  OdbcColumnVarNumeric ( const OdbcColumnVarNumeric& ) ;
  OdbcColumnVarNumeric operator = ( const OdbcColumnVarNumeric& ) ;
};

#define GEN_NUM_COLUMN_VAR(CPP_TYPE) \
  template <> class OdbcColumnVar<CPP_TYPE> : public OdbcColumnVarNumeric<CPP_TYPE> {\
  public:\
    OdbcColumnVar( unsigned int nRows = 1 ) : OdbcColumnVarNumeric<CPP_TYPE>( nRows ) {} \
  } ;

GEN_NUM_COLUMN_VAR(short)
GEN_NUM_COLUMN_VAR(unsigned short)
GEN_NUM_COLUMN_VAR(int)
GEN_NUM_COLUMN_VAR(unsigned int)
GEN_NUM_COLUMN_VAR(long)
GEN_NUM_COLUMN_VAR(unsigned long)
GEN_NUM_COLUMN_VAR(float)
GEN_NUM_COLUMN_VAR(double)
GEN_NUM_COLUMN_VAR(signed char)
GEN_NUM_COLUMN_VAR(unsigned char)
GEN_NUM_COLUMN_VAR(SQLBIGINT)
GEN_NUM_COLUMN_VAR(SQLUBIGINT)

#undef GEN_NUM_COLUMN_VAR


} // namespace odbcpp

#endif // ODBCPP_ODBCCOLUMNVAR_H
