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
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/Proxy.h"
#include "PSEvt/DataProxy.h"
#include "PSEvt/ProxyDictI.h"
#include "pdsdata/xtc/DetInfo.hh"

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
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), &typeid(const T), Pds::DetInfo(), key);
  }
  
  /**
   *  @brief Add one more proxy object to the event
   *  
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void putProxy(const boost::shared_ptr<Proxy<T> >& proxy, 
                const Pds::DetInfo& detInfo, 
                const std::string& key=std::string()) 
  {
    m_dict->put(boost::static_pointer_cast<ProxyI>(proxy), &typeid(const T), detInfo, key);
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
    m_dict->put(proxyPtr, &typeid(const T), Pds::DetInfo(), key);
  }
  
  /**
   *  @brief Add one more object to the event
   *  
   *  @param[in] data    Object to store in the event.
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  template <typename T>
  void put(const boost::shared_ptr<T>& data, 
           const Pds::DetInfo& detInfo, 
           const std::string& key=std::string()) 
  {
    boost::shared_ptr<ProxyI> proxyPtr( new DataProxy<T>(data) );
    m_dict->put(proxyPtr, &typeid(const T), detInfo, key);
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer which can be zero if object not found.
   */
  template<typename T>
  boost::shared_ptr<T> get(const std::string& key=std::string()) 
  {
    boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), Pds::DetInfo(), key);
    return boost::static_pointer_cast<T>(vptr);
  }
  
  /**
   *  @brief Get an object from event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer which can be zero if object not found.
   */
  template<typename T>
  boost::shared_ptr<T> get(const Pds::DetInfo& detInfo, const std::string& key=std::string()) 
  {
    boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), detInfo, key);
    return boost::static_pointer_cast<T>(vptr);
  }
  
  /**
   *  @brief Check if object (or proxy) of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if object or proxy exists
   */
  template <typename T>
  bool exists(const std::string& key=std::string()) 
  {
    return m_dict->exists(&typeid(const T), Pds::DetInfo(), key);
  }
  
  /**
   *  @brief Check if object (or proxy) of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if object or proxy exists
   */
  template <typename T>
  bool exists(const Pds::DetInfo& detInfo, 
              const std::string& key=std::string()) 
  {
    return m_dict->exists(&typeid(const T), detInfo, key);
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
    return m_dict->remove(&typeid(const T), Pds::DetInfo(), key);
  }
  
  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  template <typename T>
  bool remove(const Pds::DetInfo& detInfo, 
              const std::string& key=std::string()) 
  {
    return m_dict->remove(&typeid(const T), detInfo, key);
  }
  
protected:

private:

  // Data members
  boost::shared_ptr<ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEvt

#endif // PSEVT_EVENT_H
