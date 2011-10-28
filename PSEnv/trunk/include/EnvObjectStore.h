#ifndef PSENV_ENVOBJECTSTORE_H
#define PSENV_ENVOBJECTSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EnvObjectStore.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"
#include "PSEvt/DataProxy.h"
#include "PSEvt/EventKey.h"
#include "PSEvt/ProxyDictI.h"
#include "PSEvt/Source.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  @ingroup PSEnv
 *  
 *  @brief Class to store environment data objects (such as configuration
 *  or calibration) corresponding to event data objects.
 *
 *  This class is very similar to PSEvt::Event class (and is implemented 
 *  on top of the same proxy dictionary classes) but it has more specialized 
 *  interface. In particular it does not support additional string keys as
 *  it is expected that there will be only one version of the configuration
 *  or calibrations objects.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Env
 *
 *  @version \$Id: EnvObjectStore.h -1$
 *
 *  @author Andrei Salnikov
 */

class EnvObjectStore : boost::noncopyable {
public:

  /// Special class used for type-less return from get()
  struct GetResultProxy {
    
    /// Convert the result of get() call to smart pointer to object
    template<typename T>
    operator boost::shared_ptr<T>() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, std::string(), m_foundSrc);
      return boost::static_pointer_cast<T>(vptr);
    }
    
    boost::shared_ptr<PSEvt::ProxyDictI> m_dict;  ///< Proxy dictionary containing the data
    PSEvt::Source m_source;    ///< Data source address
    Pds::Src* m_foundSrc;      ///< Pointer to where to store the exact address of found object
  };

  /**
   *  @brief Standard constructor takes proxy dictionary object
   *  
   *  @param[in] dict Pointer to proxy dictionary
   */
  EnvObjectStore(const boost::shared_ptr<PSEvt::ProxyDictI>& dict) : m_dict(dict) {}

  // Destructor
  ~EnvObjectStore() {}

  /**
   *  @brief Add one more proxy object to the store
   *
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] source Source detector address.
   */
  template <typename T>
  void putProxy(const boost::shared_ptr<PSEvt::Proxy<T> >& proxy, const Pds::Src& source)
  {
    PSEvt::EventKey key(&typeid(const T), source, std::string());
    if ( m_dict->exists(key) ) {
      m_dict->remove(key);
    }
    m_dict->put(boost::static_pointer_cast<PSEvt::ProxyI>(proxy), key);
  }

  /**
   *  @brief Add one more object to the store.
   *  
   *  If there is already an object with the same type and address it 
   *  will be replaced.
   *  
   *  @param[in] data    Object to store in the event.
   *  @param[in] source Source detector address.
   */
  template <typename T>
  void put(const boost::shared_ptr<T>& data, const Pds::Src& source) 
  {
    boost::shared_ptr<PSEvt::ProxyI> proxyPtr(new PSEvt::DataProxy<T>(data));
    PSEvt::EventKey key(&typeid(const T), source, std::string());
    if ( m_dict->exists(key) ) {
      m_dict->remove(key);
    }
    m_dict->put(proxyPtr, key);
  }
  
  /**
   *  @brief Get an object from store.
   *  
   *  @param[in] source Source detector address.
   *  @return Shared pointer (or object convertible to it) which can be zero when object is not found.
   */
  GetResultProxy get(const Pds::Src& source) 
  {
    GetResultProxy pxy = { m_dict, PSEvt::Source(source) };
    return pxy;
  }

  /**
   *  @brief Get an object from store.
   *  
   *  @param[in] source Source detector address.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned 
   *                       with the exact source address of the returned object.
   *  @return Shared pointer (or object convertible to it) which can be zero when object is not found.
   */
  GetResultProxy get(const PSEvt::Source& source, Pds::Src* foundSrc=0) 
  {
    GetResultProxy pxy = { m_dict, source, foundSrc};
    return pxy;
  }

  /**
   *  @brief Get the list of keys for existing config objects
   *  
   *  @return list of the EventKey objects
   */
  std::list<PSEvt::EventKey> keys(const PSEvt::Source& source = PSEvt::Source()) const
  {
    std::list<PSEvt::EventKey> result;
    m_dict->keys(result, source);
    return result;
  }

protected:

private:

  // Data members
  boost::shared_ptr<PSEvt::ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEnv

#endif // PSENV_ENVOBJECTSTORE_H
