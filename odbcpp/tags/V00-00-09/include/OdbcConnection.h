#ifndef ODBCPP_ODBCCONNECTION_H
#define ODBCPP_ODBCCONNECTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcConnection.
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
#include "odbcpp/OdbcResult.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Connection class encapsulates all operations with connection handle.
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

class OdbcStatement ;

class OdbcConnection {
public:

  // Default constructor
  OdbcConnection ( OdbcHandle<OdbcEnv> envH, OdbcHandle<OdbcConn> connH ) ;

  // Destructor
  ~OdbcConnection () ;

  // Connect to the database or throw exception
  // Format of the connection string is the same as for SQLDriverConnect
  void connect( const std::string& connString ) ;

  // get the statement objects
  OdbcStatement statement( const std::string& q ) ;

  // get the statement objects for the list of the tables
  OdbcResultPtr tables( const std::string& catPattern,
                        const std::string& schemaPattern,
                        const std::string& tblNamePattern,
                        const std::string& tblTypePattern ) ;

  // get connection string
  const std::string& connString() const { return m_connString ; }

  // set connection attributes
  template <typename Type, int Attr>
  void setAttr ( const OdbcAttribute<Type,Attr,OdbcConnection>& attr ) ;

  // get connection attribute, maxSize is the max accepted string size,
  // not used for integer attributes
  template <typename Attr>
  typename Attr::attr_type getAttr ( unsigned int maxSize = 512 ) ;

protected:

private:

  // Data members
  OdbcHandle<OdbcEnv> m_envH ;
  boost::shared_ptr< OdbcHandle<OdbcConn> > m_connH ;
  std::string m_connString ;

};

template <typename Type, int Attr>
inline
void
OdbcConnection::setAttr ( const OdbcAttribute<Type,Attr,OdbcConnection>& attr )
{
  SQLRETURN r = SQLSetConnectAttr ( *(*m_connH), attr.attrib(), attr.getptr(), attr.size() ) ;
  OdbcStatusCheck ( r, (*m_connH) ) ;
}

// get connection attribute
template <typename Attr>
inline
typename Attr::attr_type
OdbcConnection::getAttr( unsigned int maxSize )
{
  Attr attr(this,maxSize) ;
  SQLRETURN r = SQLGetConnectAttr ( *(*m_connH), attr.attrib(), attr.setptr(), attr.setsize(), attr.setsizeptr() ) ;
  OdbcStatusCheck ( r, *m_connH ) ;
  return attr.value() ;
}

} // namespace odbcpp

#endif // ODBCPP_ODBCCONNECTION_H
