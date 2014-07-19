#ifndef PSXTCMPINPUT_SHMEMMPMASTERINPUT_H
#define PSXTCMPINPUT_SHMEMMPMASTERINPUT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemMPMasterInput.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <tr1/tuple>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "PSXtcMPInput/XtcMPMasterInputBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSXtcInput/IDatagramSource.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcMPInput {

/// @addtogroup PSXtcMPInput

/**
 *  @ingroup PSXtcMPInput
 *
 *  @brief Input module for master process in multi-process psana (shmem input only).
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ShmemMPMasterInput : public PSXtcMPInput::XtcMPMasterInputBase {
public:

  /// Constructor takes the name of the module.
  ShmemMPMasterInput (const std::string& name) ;

  // Destructor
  virtual ~ShmemMPMasterInput () ;

protected:

private:

};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_SHMEMMPMASTERINPUT_H
