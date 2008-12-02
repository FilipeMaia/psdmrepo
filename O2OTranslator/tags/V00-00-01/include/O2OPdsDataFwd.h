#ifndef O2OTRANSLATOR_O2OPDSDATAFWD_H
#define O2OTRANSLATOR_O2OPDSDATAFWD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OPdsDataFwd.
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

/**
 *  Forward declarations for the data classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: template!C++!h 4 2008-10-08 19:27:36Z salnikov $
 *
 *  @author Andrei Salnikov
 */

namespace Pds {

  // XTC-level structures
  class Dgram ;
  class Sequence ;
  class Src ;
  class DetInfo ;
  class ProcInfo ;

  // Data objects
  class WaveformV1 ;

  namespace Acqiris {
    class ConfigV1 ;
  } // namespace Acqiris

} // namespace Pds

#endif // O2OTRANSLATOR_O2OPDSDATAFWD_H
