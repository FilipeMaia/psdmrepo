#ifndef CSPAD_MOD_DATAPROXY2X2_H
#define CSPAD_MOD_DATAPROXY2X2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxy2x2.
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
 *  @brief Proxy for 2x2 Element which performs calibration.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DataProxy2x2 : public PSEvt::Proxy<Psana::CsPad2x2::ElementV1> {
public:

  // Default constructor
  DataProxy2x2 (const PSEvt::EventKey& key, PSEnv::EnvObjectStore& calibStore) ;

  // Destructor
  virtual ~DataProxy2x2 () ;

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

#endif // CSPAD_MOD_DATAPROXY2X2_H
