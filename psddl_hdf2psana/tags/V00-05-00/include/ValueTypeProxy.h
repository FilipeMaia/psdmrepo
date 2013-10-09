#ifndef PSDDL_HDF2PSANA_VALUETYPEPROXY_H
#define PSDDL_HDF2PSANA_VALUETYPEPROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ValueTypeProxy.
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

namespace psddl_hdf2psana {

/// @addtogroup psddl_hdf2psana

/**
 *  @ingroup psddl_hdf2psana
 *
 *  @brief Implementation of the event proxy for HDF value types.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename ValueType>
class ValueTypeProxy : public PSEvt::Proxy<typename ValueType::PsanaType> {
public:

  typedef typename ValueType::PsanaType PsanaType;

  // Default constructor
  ValueTypeProxy (const hdf5pp::Group& group, uint64_t idx) : m_group(group), m_idx(idx) {}

  // Destructor
  virtual ~ValueTypeProxy () {}

protected:

  /**
   *  @brief Get the data object from the proxy.
   *
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer to the data object passed to the constructor.
   */
  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict,
                                    const Pds::Src& source,
                                    const std::string& key)
  {
    if (not m_data) {
      m_data = m_hdfobj(m_group, m_idx);
    }
    return m_data;
  }

private:

  hdf5pp::Group m_group;
  uint64_t m_idx;
  ValueType m_hdfobj;
  boost::shared_ptr<PsanaType> m_data;

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_VALUETYPEPROXY_H
