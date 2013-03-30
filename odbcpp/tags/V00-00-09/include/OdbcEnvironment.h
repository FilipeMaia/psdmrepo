#ifndef ODBCPP_ODBCENVIRONMENT_H
#define ODBCPP_ODBCENVIRONMENT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcEnvironment.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcAttribute.h"
#include "odbcpp/OdbcHandle.h"
#include "odbcpp/OdbcDataSource.h"
#include "odbcpp/OdbcDriverDescription.h"
#include "odbcpp/OdbcException.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Environment class encapsulates environment handle and its operations
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

class OdbcConnection ;

class OdbcEnvironment  {
public:

  typedef std::list<OdbcDriverDescription> DriverList ;
  typedef std::list<OdbcDataSource> DsList ;

  // Default constructor
  OdbcEnvironment () ;

  // Destructor
  ~OdbcEnvironment () ;

  // get the list of drivers
  DriverList drivers() ;

  // get the list of data sources
  DsList dataSources() ;

  // Make new connection object
  OdbcConnection connection() ;

  // set environment attributes
  template <typename Type, int Attr>
  void setAttr ( const OdbcAttribute<Type,Attr,OdbcEnvironment>& attr ) ;

  // get environment attribute, maxSize is the max accepted string size,
  // not used for integer attributes
  template <typename Attr>
  typename Attr::attr_type getAttr ( unsigned int maxSize = 512 ) ;

protected:

private:

  // Data members
  OdbcHandle<OdbcEnv> m_envH ;

};

template <typename Type, int Attr>
inline
void
OdbcEnvironment::setAttr ( const OdbcAttribute<Type,Attr,OdbcEnvironment>& attr )
{
  SQLRETURN r = SQLSetEnvAttr ( *m_envH, attr.attrib(), attr.getptr(), attr.size() ) ;
  OdbcStatusCheck ( r, m_envH ) ;
}

// get environment attribute
template <typename Attr>
inline
typename Attr::attr_type
OdbcEnvironment::getAttr( unsigned int maxSize )
{
  Attr attr(this,maxSize) ;
  SQLRETURN r = SQLGetEnvAttr ( *m_envH, attr.attrib(), attr.setptr(), attr.setsize(), attr.setsizeptr() ) ;
  OdbcStatusCheck ( r, m_envH ) ;
  return attr.value() ;
}

} // namespace odbcpp

#endif // ODBCPP_ODBCENVIRONMENT_H
