#ifndef PSEVT_EVENT_H
#define PSEVT_EVENT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Event.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventKey.h"
#include "PSEvt/Proxy.h"
#include "PSEvt/DataProxy.h"
#include "PSEvt/ProxyDictI.h"
#include "PSEvt/Source.h"
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  @defgroup PSEvt  PSEvt package
 *  
 *  @brief PSEvt package contains classes which provide storage and 
 *  access to event data in the context of psana framework.
 *  
 *  The core of the package is the proxy dictionary classes which allow 
 *  storage of the arbitrary types. Dictionaries to not store data 
 *  directly, instead they store proxy objects which can either contain
 *  data objects or implement algorithm to generate data objects when
 *  necessary.
 *  
 *  Main user interface to this package is the Event class which is a 
 *  wrapper for proxy dictionary providing more user-friendly interface.
 */

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Class which manages event data in psana framework.
 *  
 *  This class is a user-friendly interface to proxy dictionary object. 
 *  It provides a number of put() and get() methods to store/retrieve 
 *  arbitrarily typed data.
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

class Event {
public:

  /// Special class used for type-less return from get()
  struct GetResultProxy {
    
    /// Convert the result of Event::get() call to smart pointer to data object
    template<typename T>
    operator boost::shared_ptr<T>() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, m_key, m_foundSrc);
      return boost::static_pointer_cast<T>(vptr);
    }
    
    boost::shared_ptr<ProxyDictI> m_dict; ///< Proxy dictionary containing the data
    Source m_source;         ///< Data source address
    std::string m_key;       ///< String key
    Pds::Src* m_foundSrc;    ///< Pointer to where to store the exact address of found object
  };
  
  
  /**
   *  @brief Standard constructor takes proxy dictionary object
   *  
   *  @param[in] dict Pointer to proxy dictionary
   */
  Event(const boost::shared_ptr<ProxyDictI>& dict) : m_dict(dict) {}

  //Destructor
  ~Event () {}

  /**
   *  @brief Add one more proxy object to the event
   *  
   *  @param[in] proxy  Proxy object for type T.
   *  @param[in] key    Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void putProxy(const boost::shared_ptr<Proxy<T> >& proxy, const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), EventKey::noSource(), key);
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), evKey);
  }
  
  /**
   *  @brief Add one more proxy object to the event
   *  
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void putProxy(const boost::shared_ptr<Proxy<T> >& proxy, 
                const Pds::Src& source, 
                const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), source, key);
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), evKey);
  }
  
  /**
   *  @brief Add one more object to the event
   *  
   *  @param[in] data   Object to store in the event.
   *  @param[in] key    Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void put(const boost::shared_ptr<T>& data, const std::string& key=std::string()) 
  {
    boost::shared_ptr<ProxyI> proxyPtr(boost::make_shared<DataProxy<T> >(data) );
    EventKey evKey(&typeid(const T), EventKey::noSource(), key);
    m_dict->put(proxyPtr, evKey);
  }
  
  /**
   *  @brief Add one more object to the event
   *  
   *  @param[in] data    Object to store in the event.
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void put(const boost::shared_ptr<T>& data, 
           const Pds::Src& source, 
           const std::string& key=std::string()) 
  {
    boost::shared_ptr<ProxyI> proxyPtr(boost::make_shared<DataProxy<T> >(data) );
    EventKey evKey(&typeid(const T), source, key);
    m_dict->put(proxyPtr, evKey);
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned 
   *                       with the exact source address of the returned object.
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const std::string& key=std::string(), Pds::Src* foundSrc=0)
  {
    GetResultProxy pxy = {m_dict, Source(Source::null), key, foundSrc};
    return pxy;
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned 
   *                       with the exact source address of the returned object (must 
   *                       be the same as source)
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const Pds::Src& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) 
  {
    GetResultProxy pxy = {m_dict, Source(source), key, foundSrc};
    return pxy;
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned 
   *                       with the exact source address of the returned object.
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) 
  {
    GetResultProxy pxy = {m_dict, source, key, foundSrc};
    return pxy;
  }
  
  /**
   *  @brief Check if object (or proxy) of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if object or proxy exists
   */
  template <typename T>
  bool exists(const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), Pds::Src(), key);
    return m_dict->exists(evKey);
  }
  
  /**
   *  @brief Check if object (or proxy) of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if object or proxy exists
   */
  template <typename T>
  bool exists(const Pds::Src& source, 
              const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), source, key);
    return m_dict->exists(evKey);
  }
  
  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  template <typename T>
  bool remove(const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), Pds::Src(), key);
    return m_dict->remove(evKey);
  }
  
  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  template <typename T>
  bool remove(const Pds::Src& source, 
              const std::string& key=std::string()) 
  {
    EventKey evKey(&typeid(const T), source, key);
    return m_dict->remove(evKey);
  }
  
  /**
   *  @brief Get the list of event keys defined in event matching given source
   *  
   *  @param[in]  source matching source address
   *  @return list of the EventKey objects
   */
  std::list<EventKey> keys(const Source& source = Source()) const
  {
    std::list<EventKey> result;
    m_dict->keys(result, source);
    return result;
  }
  
protected:

private:

  // Data members
  boost::shared_ptr<ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEvt

#endif // PSEVT_EVENT_H
