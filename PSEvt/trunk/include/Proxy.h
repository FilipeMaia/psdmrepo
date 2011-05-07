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
#include "PSEvt/ProxyI.h"

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
 *  @brief Interface class for type-safe proxy classes. 
 *  
 *  Proxy dictionary stores proxies of type ProxyI which are not 
 *  type-safe (they work with void pointers). To make code safer
 *  this class implements ProxyI interface and introduces type-safe 
 *  method to generate typed data. User-level interface should use this
 *  type instead of ProxyI.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see ProxyI
 *
 *  @version \$Id$
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
   *  @brief Get untyped object from the proxy.
   *  
   *  The parameters passed to the proxy can be used by the proxy 
   *  to find additional information from the same (or different)
   *  detector. 
   *  
   *  This is implementation of ProxyI interface which forwards 
   *  call to the type-safe method getTypedImpl().
   *  
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl(ProxyDictI* dict,
                                          const Pds::Src& source, 
                                          const std::string& key)
  {
    return boost::static_pointer_cast<void>(getTypedImpl(dict, source, key));
  }

  /**
   *  @brief Get the correctly-typed object from the proxy.
   *    
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of the correct type.
   */
  virtual boost::shared_ptr<T> getTypedImpl(ProxyDictI* dict,
                                            const Pds::Src& source, 
                                            const std::string& key) = 0;

private:

  // Data members
};

} // namespace PSEvt

#endif // PSEVT_PROXY_H
