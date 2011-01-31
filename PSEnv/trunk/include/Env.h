#ifndef PSENV_ENV_H
#define PSENV_ENV_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Env.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  @brief Class representing an environment object for psana jobs.
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

class Env  {
public:

  // Default constructor
  Env () ;

  // Destructor
  virtual ~Env () ;

protected:

private:

  // Data members

};

} // namespace PSEnv

#endif // PSENV_ENV_H
