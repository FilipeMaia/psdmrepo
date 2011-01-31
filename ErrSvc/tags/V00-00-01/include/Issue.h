#ifndef ERRSVC_ISSUE_H
#define ERRSVC_ISSUE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Issue.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include <exception>
#include <iosfwd>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ErrSvc/Context.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ErrSvc {

/**
 *  @brief Base class for other error classes.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Issue : public std::exception {
public:

  // Default constructor
  Issue (const Context& ctx, const std::string& message) ;

  // Destructor
  virtual ~Issue () throw() ;

  // Implements std::exception::what()
  virtual const char* what() const throw();

  /// Returns context
  const Context& context() const { return m_ctx; }
  
  /// Returns original message
  const std::string& message() const { return m_message; }
  
protected:

private:

  // Data members
  Context m_ctx;
  std::string m_message;
  std::string m_fullMessage;  /// Message string plus context

};

/// Stream insertion operator
std::ostream&
operator<<(std::ostream& out, const Issue& issue);

} // namespace ErrSvc

#endif // ERRSVC_ISSUE_H
