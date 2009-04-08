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
 *  @version $Id$
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

  namespace Acqiris {
    class ConfigV1 ;
    class DataDescV1 ;
  } // namespace Acqiris

  namespace Camera {

    class FrameFexConfigV1 ;
    class FrameV1 ;
    class TwoDGaussianV1 ;

  } // namespace Camera

  namespace EvrData {

    class ConfigV1 ;

  } // namespace EvrData

  namespace Opal1k {

    class ConfigV1 ;

  } // namespace Opal1k

} // namespace Pds



#endif // O2OTRANSLATOR_O2OPDSDATAFWD_H
