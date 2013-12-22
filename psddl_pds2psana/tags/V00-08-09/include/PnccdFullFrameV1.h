#ifndef PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1_H
#define PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdFullFrameV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/pnccd.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/// @addtogroup psddl_pds2psana

/**
 *  @ingroup psddl_pds2psana
 *
 *  @brief Special implementation of PNCCD::FullFrameV1 for psana.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class PnccdFullFrameV1 : public Psana::PNCCD::FullFrameV1 {
public:

  // Default constructor
  PnccdFullFrameV1 (const Psana::PNCCD::FramesV1& frames) ;

  // Destructor
  virtual ~PnccdFullFrameV1 () ;

protected:

  /** Special values */
  virtual uint32_t specialWord() const;
  /** Frame number */
  virtual uint32_t frameNumber() const;
  /** Most significant part of timestamp */
  virtual uint32_t timeStampHi() const;
  /** Least significant part of timestamp */
  virtual uint32_t timeStampLo() const;
  /** Full frame data, image size is 1024x1024. */
  virtual ndarray<const uint16_t, 2> data() const;

private:

  uint32_t  _specialWord;   /**< Special values */
  uint32_t  _frameNumber;   /**< Frame number */
  uint32_t  _timeStampHi;   /**< Most significant part of timestamp */
  uint32_t  _timeStampLo;   /**< Least significant part of timestamp */
  uint16_t  _data[1024][1024];  /**< Full frame data, image size is 1024x1024. */

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1_H
