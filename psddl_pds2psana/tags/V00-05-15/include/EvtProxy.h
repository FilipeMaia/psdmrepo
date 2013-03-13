#ifndef PSDDL_PDS2PSANA_EVTPROXY_H
#define PSDDL_PDS2PSANA_EVTPROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtProxy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/Proxy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

  
// Factory methods to create new object with different constructor signatures
template <typename PDS2PSType, typename XTCType, bool UseSize>
struct EvtProxyFactory {};

template <typename PDS2PSType, typename XTCType>
struct EvtProxyFactory<PDS2PSType, XTCType, false> { 
  static PDS2PSType* create(const boost::shared_ptr<XTCType> xtcObj, size_t /*xtcSize*/) {
    return new PDS2PSType(xtcObj);
  }
};

template <typename PDS2PSType, typename XTCType>
struct EvtProxyFactory<PDS2PSType, XTCType, true> { 
  static PDS2PSType* create(const boost::shared_ptr<XTCType> xtcObj, size_t xtcSize) {
    return new PDS2PSType(xtcObj, xtcSize);
  }
};

  
/**
 *  @brief Implementation of the proxy interface for the XTC data object 
 *  that does not need config object.
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

template <typename PSType, typename PDS2PSType, typename XTCType, bool UseSize=false>
class EvtProxy : public PSEvt::Proxy<PSType> {
public:

  // Default constructor
  EvtProxy (const boost::shared_ptr<XTCType>& xtcObj, size_t xtcSize) 
    : m_xtcObj(xtcObj), m_xtcSize(xtcSize) {}

  // Destructor
  virtual ~EvtProxy () {}

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
      typedef EvtProxyFactory<PDS2PSType, XTCType, UseSize> Factory;
      m_psObj.reset(Factory::create(m_xtcObj, m_xtcSize));
    }
    return m_psObj;
  }
  
private:

  
  // Data members
  boost::shared_ptr<XTCType> m_xtcObj;
  size_t m_xtcSize;
  boost::shared_ptr<PSType> m_psObj;
  
};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_EVTPROXY_H
