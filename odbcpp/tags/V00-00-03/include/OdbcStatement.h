#ifndef ODBCPP_ODBCSTATEMENT_H
#define ODBCPP_ODBCSTATEMENT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcStatement.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcAttribute.h"
#include "odbcpp/OdbcException.h"
#include "odbcpp/OdbcHandle.h"
#include "odbcpp/OdbcParam.h"
#include "odbcpp/OdbcResult.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  ODBC statement type encapsulates all operations with statement handle
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

class OdbcStatement  {
public:

  // Default constructor
  OdbcStatement ( OdbcHandle<OdbcStmt> stmtH ) ;

  // Destructor
  ~OdbcStatement () ;

  // execute statement
  OdbcResultPtr execute() ;

  // bind single parameter
  template <typename T>
  void bindParam ( unsigned i, OdbcParam<T>& param ) ;

  // unbind all parameters
  void unbindParams () ;

  // set statement attributes
  template <typename Type, int Attr>
  void setAttr ( const OdbcAttribute<Type,Attr,OdbcStatement>& attr ) ;

  // get statement attribute, maxSize is the max accepted string size,
  // not used for integer attributes
  template <typename Attr>
  typename Attr::attr_type getAttr ( unsigned int maxSize = 512 ) ;

protected:

private:

  // Data members
  OdbcHandle<OdbcStmt> m_stmtH ;

};

// bind single parameter
template <typename T>
inline
void
OdbcStatement::bindParam ( unsigned i, OdbcParam<T>& param )
{
  SQLRETURN r = SQLBindParameter(
       *m_stmtH,
       i,
       param.inputOutputType(),
       param.valueType(),
       param.parameterType(),
       param.columnSize(),
       param.decimalDigits(),
       param.parameterValuePtr(),
       param.bufferLength(),
       param.strLenOrIndPtr() );
  OdbcStatusCheck ( r, m_stmtH ) ;
}


template <typename Type, int Attr>
inline
void
OdbcStatement::setAttr ( const OdbcAttribute<Type,Attr,OdbcStatement>& attr )
{
  SQLRETURN r = SQLSetStmtAttr ( *m_stmtH, attr.attrib(), attr.getptr(), attr.size() ) ;
  OdbcStatusCheck ( r, m_stmtH ) ;
}

// get connection attribute
template <typename Attr>
inline
typename Attr::attr_type
OdbcStatement::getAttr( unsigned int maxSize )
{
  Attr attr(this,maxSize) ;
  SQLRETURN r = SQLGetStmtAttr ( *m_stmtH, attr.attrib(), attr.setptr(), attr.setsize(), attr.setsizeptr() ) ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  return attr.value() ;
}

} // namespace odbcpp

#endif // ODBCPP_ODBCSTATEMENT_H
