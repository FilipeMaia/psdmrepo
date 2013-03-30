#ifndef ODBCPP_ODBCDRIVERDESCRIPTION_H
#define ODBCPP_ODBCDRIVERDESCRIPTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcDriverDescription.
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
 *  Encapsulation of the driver desription
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

class OdbcDriverDescription {
public:

  // Default constructor
  OdbcDriverDescription ( const std::string& driver, const std::string& attr )
    : m_driver(driver), m_attr(attr) {}

  // Destructor
  ~OdbcDriverDescription () {}

  const std::string& driver() const { return m_driver ; }
  const std::string& attr() const { return m_attr ; }

private:

  // Data members
  std::string m_driver ;
  std::string m_attr ;

};

std::ostream&
operator << ( std::ostream& out, const OdbcDriverDescription& d ) ;

} // namespace odbcpp

#endif // ODBCPP_ODBCDRIVERDESCRIPTION_H
