#ifndef PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1PROXY_H
#define PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1PROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdFullFrameV1Proxy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/Proxy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Xtc.hh"
#include "psddl_psana/pnccd.ddl.h"

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
 *  @brief Special proxy class for PNCCD::FullFrameV1
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class PnccdFullFrameV1Proxy : public PSEvt::Proxy<Psana::PNCCD::FullFrameV1> {
public:

  // Default constructor
  PnccdFullFrameV1Proxy (const boost::shared_ptr<Pds::Xtc>& xtcObj) ;

  // Destructor
  virtual ~PnccdFullFrameV1Proxy () ;

protected:

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<Psana::PNCCD::FullFrameV1>
  getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  boost::shared_ptr<Psana::PNCCD::FullFrameV1> m_psObj;

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_PNCCDFULLFRAMEV1PROXY_H
