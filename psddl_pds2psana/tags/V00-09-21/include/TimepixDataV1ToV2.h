#ifndef PSDDL_PDS2PSANA_TIMEPIXDATAV1TOV2_H
#define PSDDL_PDS2PSANA_TIMEPIXDATAV1TOV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV1ToV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/timepix.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/psddl/timepix.ddl.h"

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
 *  @brief Special implementation of Psana::Timepix::DataV2 interface
 *  which can be constructed from Pds::Timepix::DataV1 class.
 *
 *  This special implementation is needed to re-shuffle data inside
 *  DataV1 class which is not usable directly without re-shuffling.
 *  Note that we use DataV2 interface instead of dataV1 to make this
 *  thing compatible with other places which do re-shuffling by
 *  converting DataV1 class into DataV2.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class TimepixDataV1ToV2 : public Psana::Timepix::DataV2 {
public:

  typedef Pds::Timepix::DataV1 XtcType;
  typedef Psana::Timepix::DataV2 PsanaType;

  // Default constructor
  TimepixDataV1ToV2(const boost::shared_ptr<const XtcType>& xtcPtr);

  // Destructor
  virtual ~TimepixDataV1ToV2 () ;

  virtual uint16_t width() const;
  virtual uint16_t height() const;
  virtual uint32_t timestamp() const;
  virtual uint16_t frameCounter() const;
  virtual uint16_t lostRows() const;
  virtual ndarray<const uint16_t, 2> data() const;
  virtual uint32_t depth() const;
  virtual uint32_t depth_bytes() const;
  virtual uint32_t data_size() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }

private:

  boost::shared_ptr<const Pds::Timepix::DataV1> m_xtcObj;
  mutable uint16_t* m_data;

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_TIMEPIXDATAV1TOV2_H
