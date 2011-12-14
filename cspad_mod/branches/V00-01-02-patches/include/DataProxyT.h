#ifndef CSPAD_MOD_DATAPROXYT_H
#define CSPAD_MOD_DATAPROXYT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxyT.
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
#include "cspad_mod/DataT.h"
#include "cspad_mod/ElementT.h"
#include "psddl_psana/cspad.ddl.h"
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
 *  @brief Proxy for DataT/ElementT which performs calibration.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename DataType, typename ElemType>
class DataProxyT : public PSEvt::Proxy<typename DataType::IfaceType> {
public:

  typedef typename DataType::IfaceType DataIfaceType;
  typedef typename ElemType::IfaceType ElemIfaceType;

  // Default constructor
  DataProxyT (const PSEvt::EventKey& key, PSEnv::EnvObjectStore& calibStore) ;

  // Destructor
  virtual ~DataProxyT () ;

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<DataIfaceType>
  getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

protected:

private:

  // Data members
  PSEvt::EventKey m_key;
  PSEnv::EnvObjectStore& m_calibStore;
  boost::shared_ptr<DataType> m_data;

};

typedef DataProxyT<cspad_mod::DataV1, cspad_mod::ElementV1> DataProxyV1;
typedef DataProxyT<cspad_mod::DataV2, cspad_mod::ElementV2> DataProxyV2;

} // namespace cspad_mod

#endif // CSPAD_MOD_DATAPROXYT_H
