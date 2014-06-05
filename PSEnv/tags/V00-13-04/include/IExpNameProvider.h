#ifndef PSENV_IEXPNAMEPROVIDER_H
#define PSENV_IEXPNAMEPROVIDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IExpNameProvider.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

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

namespace PSEnv {

/**
 *  @ingroup PSEnv
 *
 *  @brief CLass which defines an interface for obtaining  instrument and
 *  experiment names.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class IExpNameProvider : boost::noncopyable {
public:

  // Destructor
  virtual ~IExpNameProvider() {}
  
  /// Returns instrument name
  virtual const std::string& instrument() const = 0;

  /// Returns experiment name
  virtual const std::string& experiment() const = 0;

  /// Returns experiment number or 0
  virtual unsigned expNum() const = 0;

protected:

  // Constructor
  IExpNameProvider () {}

private:

};

} // namespace PSEnv

#endif // PSENV_IEXPNAMEPROVIDER_H
