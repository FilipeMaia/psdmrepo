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
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  Class representing event data in psana,
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

class Event {
public:

  // Special class used for type-less return from get()
  struct GetResultProxy {
    
    template<typename T>
    operator boost::shared_ptr<T>() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, m_key);
      return boost::static_pointer_cast<T>(vptr);
    }
    
    boost::shared_ptr<ProxyDictI> m_dict;
    Pds::Src m_source;
    std::string m_key;
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
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), &typeid(const T), Pds::Src(), key);
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
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), &typeid(const T), source, key);
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
    boost::shared_ptr<ProxyI> proxyPtr( new DataProxy<T>(data) );
    m_dict->put(proxyPtr, &typeid(const T), Pds::Src(), key);
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
    boost::shared_ptr<ProxyI> proxyPtr( new DataProxy<T>(data) );
    m_dict->put(proxyPtr, &typeid(const T), source, key);
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const std::string& key=std::string()) 
  {
    GetResultProxy pxy = {m_dict, Pds::Src(), key};
    return pxy;
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const Pds::Src& source, const std::string& key=std::string()) 
  {
    GetResultProxy pxy = {m_dict, source, key};
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
    return m_dict->exists(&typeid(const T), Pds::Src(), key);
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
    return m_dict->exists(&typeid(const T), source, key);
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
    return m_dict->remove(&typeid(const T), Pds::Src(), key);
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
    return m_dict->remove(&typeid(const T), source, key);
  }
  
  /**
   *  @brief Get the list of event keys defined in event
   *  
   *  @return list of the EventKey objects
   */
  std::list<EventKey> keys() const
  {
    std::list<EventKey> result;
    m_dict->keys(result);
    return result;
  }
  
protected:

private:

  // Data members
  boost::shared_ptr<ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEvt

#endif // PSEVT_EVENT_H
