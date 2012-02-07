#ifndef CSPAD_MOD_DATAPROXYMINI_H
#define CSPAD_MOD_DATAPROXYMINI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxyMini.
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
#include "psddl_psana/cspad2x2.ddl.h"
#include "PSEnv/EnvObjectStore.h"
#include "PSEvt/EventKey.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Proxy for MiniElement which performs calibration.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DataProxyMini : public PSEvt::Proxy<Psana::CsPad2x2::ElementV1> {
public:

  // Default constructor
  DataProxyMini (const PSEvt::EventKey& key, PSEnv::EnvObjectStore& calibStore) ;

  // Destructor
  virtual ~DataProxyMini () ;

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<Psana::CsPad2x2::ElementV1>
  getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

protected:

private:

  // Data members
  PSEvt::EventKey m_key;
  PSEnv::EnvObjectStore& m_calibStore;
  boost::shared_ptr<Psana::CsPad2x2::ElementV1> m_data;

};

} // namespace cspad_mod

#endif // CSPAD_MOD_DATAPROXYMINI_H
