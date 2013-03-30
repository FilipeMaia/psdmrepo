#ifndef ODBCPP_ODBCDATASOURCE_H
#define ODBCPP_ODBCDATASOURCE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcDataSource.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Encapsulation of the data source description: DSN and driver name
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

class OdbcDataSource  {
public:

  // Default constructor
  OdbcDataSource ( const std::string& dsn, const std::string& driver )
    : m_dsn(dsn), m_driver(driver) {}

  // Destructor
  ~OdbcDataSource () {}

  const std::string& dsn() const { return m_dsn ; }
  const std::string& driver() const { return m_driver ; }

private:

  // Data members
  std::string m_dsn ;
  std::string m_driver ;

};

std::ostream&
operator << ( std::ostream& out, const OdbcDataSource& ds ) ;

} // namespace odbcpp

#endif // ODBCPP_ODBCDATASOURCE_H
