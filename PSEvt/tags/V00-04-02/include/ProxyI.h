#ifndef PSEVT_PROXYI_H
#define PSEVT_PROXYI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProxyI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

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
 *  @brief Class defining interface for all proxy classes.
 *  
 *  This is the interface used by the proxy dictionary which stores
 *  the objects of this type. This interface is untyped, as it returns 
 *  pointer to void. It should not be exposed to user code.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see ProxyDictI
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class ProxyI : boost::noncopyable {
public:

  // Destructor
  virtual ~ProxyI () {}

  /**
   *  @brief Get the object from the proxy.
   *  
   *  The parameters passed to the proxy can be used by the proxy 
   *  to find additional information from the same (or different)
   *  detector. 
   *  
   *  By default the request is forwarded to the virtual method getImp() 
   *  (customization point) but there is a possibility to do something 
   *  else too if needed.
   *  
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of void type.
   */
  boost::shared_ptr<void> get(ProxyDictI* dict,
                              const Pds::Src& source, 
                              const std::string& key)
  {
    return this->getImpl(dict, source, key);
  }
  
protected:

  // Default constructor
  ProxyI () {}

  /**
   *  @brief Get the object from the proxy.
   *  
   *  The parameters passed to the proxy can be used by the proxy 
   *  to find additional information from the same (or different)
   *  detector. 
   *  
   *  @param[in] dict    Proxy dictionary containing this proxy.
   *  @param[in] source Detector address information
   *  @param[in] key     String key, additional key supplied by user.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl(ProxyDictI* dict,
                                          const Pds::Src& source, 
                                          const std::string& key) = 0;

private:

};

} // namespace PSEvt

#endif // PSEVT_PROXYI_H
