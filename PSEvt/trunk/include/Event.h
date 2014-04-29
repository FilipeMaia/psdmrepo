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
#include <boost/enable_shared_from_this.hpp>
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
#include "ndarray/ndarray.h"
#include "MsgLogger/MsgLogger.h"

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

class Event : public boost::enable_shared_from_this<Event>, boost::noncopyable {
public:

  /// Special class used for type-less return from get()
  struct GetResultProxy {
    
    /// Convert the result of Event::get() call to smart pointer to data object
    template<typename T>
    operator boost::shared_ptr<T>() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, m_key, m_foundSrc);
      return boost::static_pointer_cast<T>(vptr);
    }

    /// specializiation to help users diagnose problems with using a non const ndarray template 
    /// argument when a const argument was intended
    template<typename T, unsigned NDim>
    operator boost::shared_ptr< ndarray<T,NDim> >() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const ndarray<T, NDim>), m_source, m_key, m_foundSrc);
      if (not vptr and m_dict->get(&typeid(const ndarray<const T, NDim>), m_source, m_key, 0)) {
        MsgLog("Event::get",warning,"Event::get - requested ndarray<T,R> *not* present *but* ndarray<const T,R> is for"
               << " src=" << m_source << " key=" << m_key);
      }
      return boost::static_pointer_cast< ndarray<T,NDim> >(vptr);
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
  explicit Event(const boost::shared_ptr<ProxyDictI>& dict) : m_dict(dict) {}

  //Destructor
  ~Event () {}

  /**
   *  @brief Add one more proxy object to the event
   *  
   *  @param[in] proxy  Proxy object for type T.
   *  @param[in] key    Optional key to distinguish different objects of the same type.
   *
   *  @throw ExceptionDuplicateKey
   *  @throw ExceptionNoAliasMap
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
   *
   *  @throw ExceptionDuplicateKey
   *  @throw ExceptionNoAliasMap
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
   *
   *  @throw ExceptionDuplicateKey
   *  @throw ExceptionNoAliasMap
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
   *
   *  @throw ExceptionDuplicateKey
   *  @throw ExceptionNoAliasMap
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
   *  This method finds and returns object in an event which is not associated
   *  with any detector device (this is why it does not have source argument).
   *  It can be used for example to obtain EventId object or other similar types
   *  of data.
   *
   *  Note that if you pass an std::string to get() then this method will be
   *  called even if string may look like device address. Always use Source
   *  class as an argument to get() to locate detector data.
   *
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer which can be zero if object not found.
   */
  GetResultProxy get(const std::string& key=std::string())
  {
    GetResultProxy pxy = {m_dict, Source(Source::null), key, (Pds::Src*)(0)};
    return pxy;
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  Find and return data object which was produced by specific device. Device is
   *  specified as a source object of type Pds::Src. This overloaded function should
   *  be used if the source is known exactly, for example when Pds::Src object is
   *  returned via foundSrc pointer from previous call to get() method.
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
   *  Find and return data object which was produced by device. This method accepts
   *  Source object which allows approximate specification of the device addresses.
   *  If specified address matches more than one device in the event then one arbitrary
   *  object is returned. The foundSrc argument can be used to obtain exact address
   *  of a returned object.
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
   *  is there but does not ask proxy to do any real work. It is not guaranteed
   *  that get() will return any data even if exists() returns true, proxy may
   *  decide that its corresponding data does not exits.
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
   *  is there but does not ask proxy to do any real work. It is not guaranteed
   *  that get() will return any data even if exists() returns true, proxy may
   *  decide that its corresponding data does not exits.
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
  
  /**
   *  @brief Get access to proxy dictionary.
   *
   *  This method exposes underlying proxy dictionary object. It should
   *  not be used by ordinary clients but it could be useful for code
   *  which implements additional services based on event (such as
   *  Python wrappers).
   */
  const boost::shared_ptr<ProxyDictI>& proxyDict() const { return m_dict; }

protected:

private:

  // Data members
  boost::shared_ptr<ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEvt

#endif // PSEVT_EVENT_H
