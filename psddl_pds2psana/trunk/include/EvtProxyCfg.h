#ifndef PSDDL_PDS2PSANA_EVTPROXYCFG_H
#define PSDDL_PDS2PSANA_EVTPROXYCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtProxyCfg.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/**
 *  @brief Implementation of the proxy interface for the XTC data object 
 *  which need (one) config object.
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

template <typename PSType, typename PDS2PSType, typename XTCType, typename XTCConfigType>
class EvtProxyCfg : public PSEvt::Proxy<PSType> {
public:

  // Default constructor
  EvtProxyCfg (const boost::shared_ptr<Pds::Xtc>& xtcObj,
      const boost::shared_ptr<XTCConfigType>& cfgObj)
    : m_xtcObj(xtcObj), m_cfgObj(cfgObj) {}

  // Destructor
  virtual ~EvtProxyCfg () {}

protected:

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *    
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<PSType> getTypedImpl(PSEvt::ProxyDictI* dict,
                                            const Pds::Src& source, 
                                            const std::string& key)
  {
    if (not m_psObj.get()) {

      // decompress it if needed
      if (m_xtcObj->contains.compressed()) {
        m_xtcObj = Pds::CompressedXtc::uncompress(*m_xtcObj);
      }

      // get pointer to data
      boost::shared_ptr<XTCType> xptr(m_xtcObj, (XTCType*)(m_xtcObj->payload()));

      m_psObj.reset(new PDS2PSType(xptr, m_cfgObj));
    }
    return m_psObj;
  }
  
private:

  // Data members
  boost::shared_ptr<Pds::Xtc> m_xtcObj;
  boost::shared_ptr<XTCConfigType> m_cfgObj;
  boost::shared_ptr<PSType> m_psObj;

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_EVTPROXYCFG_H
