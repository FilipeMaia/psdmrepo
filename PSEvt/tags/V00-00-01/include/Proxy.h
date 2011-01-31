#ifndef PSEVT_PROXY_H
#define PSEVT_PROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Proxy.
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
#include "PsEvt/ProxyI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/DetInfo.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PsEvt {
  class ProxyDictI;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsEvt {

/**
 *  @brief Interface class for type-save proxy classes. 
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

template <typename T>
class Proxy : public ProxyI {
public:

  // Destructor
  virtual ~Proxy () {}

protected:

  // Default constructor
  Proxy () {}

  /**
   *  @brief Get the object from the proxy.
   *  
   *  The parameters passed to the proxy can be used by the proxy 
   *  to find additional information from the same (or different)
   *  detector. 
   *  
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] detInfo Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl(ProxyDictI* dict,
                                          const Pds::DetInfo& detInfo, 
                                          const std::string& key)
  {
    return boost::static_pointer_cast<void>(getTypedImpl(dict, detInfo, key));
  }

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *    
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] detInfo Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<T> getTypedImpl(ProxyDictI* dict,
                                            const Pds::DetInfo& detInfo, 
                                            const std::string& key) = 0;

private:

  // Data members
};

} // namespace PsEvt

#endif // PSEVT_PROXY_H
