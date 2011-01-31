#ifndef ERRSVC_CONTEXT_H
#define ERRSVC_CONTEXT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Context.
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

// evil macro
#define ERR_LOC ErrSvc::Context( __FILE__, __LINE__, __func__ )


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ErrSvc {

/**
 *  @brief Class describing the context where the issue happened.
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

class Context  {
public:

  /**
   *  @brief Constructor takes location arguments
   */ 
  Context( const char* file, int line, const char* func ) ;

  /// Return file name
  const std::string& file() const { return m_file; }
  
  /// Return line number
  int line() const { return m_line; }
  
  /// Return function name
  const std::string& function() const { return m_func; }
  
protected:

private:

  // Data members
  std::string m_file;
  std::string m_func;
  int m_line;
  
};

/// Stream insertion operator
std::ostream&
operator<<(std::ostream& out, const Context& ctx);

} // namespace ErrSvc

#endif // ERRSVC_CONTEXT_H
