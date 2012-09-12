#ifndef PSEVT_DATAPROXY_H
#define PSEVT_DATAPROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataProxy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/Proxy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSEvt {
  class ProxyDictI;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Implementation of proxy object which keeps a pointer to real object.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Proxy
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

template <typename T>
class DataProxy : public Proxy<T> {
public:

  /// Constructor takes a smart pointer to the data object.
  DataProxy (const boost::shared_ptr<T>& data) : m_data(data) {}

  // Destructor
  virtual ~DataProxy () {}

protected:

  /**
   *  @brief Get the data object from the proxy.
   *    
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer to the data object passed to the constructor.
   */
  virtual boost::shared_ptr<T> getTypedImpl(ProxyDictI* dict,
                                            const Pds::Src& source, 
                                            const std::string& key)
  {
    return m_data;
  }

private:

  // Data members
  boost::shared_ptr<T> m_data; ///< Stored pointer to the data object.
};

} // namespace PSEvt

#endif // PSEVT_DATAPROXY_H
