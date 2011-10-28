#ifndef PSEVT_PROXYDICTI_H
#define PSEVT_PROXYDICTI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProxyDictI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <typeinfo>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"
#include "PSEvt/EventKey.h"
#include "PSEvt/ProxyI.h"
#include "PSEvt/Source.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Class defining an interface for all proxy dictionary classes.
 *  
 *  The client-side interface of this class is non-virtual and it 
 *  forwards every class to virtual methods which define customization
 *  oints to be implemented in subclasses.
 *  
 *  Proxy dictionary stores proxy objects of type ProxyI which represent
 *  actual objects and know how to create or retrieve an object when
 *  requested. Proxy collection is indexed by the EventKey which identifies
 *  object data type, data source address, and additional string key.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see ProxyDict
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class ProxyDictI : boost::noncopyable {
public:

  // Destructor
  virtual ~ProxyDictI () {}

  /**
   *  @brief Add one more proxy object to the dictionary.
   *  
   *  By default the request is forwarded to the virtual method  
   *  (customization point) but there is a possibility to do something 
   *  else too if needed.

   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] key     Event key for the data object.
   */
  void put( const boost::shared_ptr<ProxyI>& proxy, const EventKey& key ) 
  {
    this->putImpl(proxy, key);
  }

  /**
   *  @brief Get an object from event
   * 
   *  @param[in] typeinfo  Dynamic type info object
   *  @param[in] source    Source detector address.
   *  @param[in] key       Optional key to distinguish different objects of the same type.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned
   *                       with the exact source address of the returned object.
   *  @return Shared pointer of void type.
   */
  boost::shared_ptr<void> get( const std::type_info* typeinfo, 
                               const Source& source, 
                               const std::string& key,
                               Pds::Src* foundSrc)
  {
    return this->getImpl(typeinfo, source, key, foundSrc);
  }


  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] key     Event key for the data object.
   *  @return true if proxy exists
   */
  bool exists(const EventKey& key)
  {
    return this->existsImpl(key);
  }

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] key     Event key for the data object.
   *  @return false if object did not exist before this call
   */
  bool remove(const EventKey& key)
  {
    return this->removeImpl(key);
  }

  /**
   *  @brief Get the list of event keys defined in event
   *  
   *  @param[in]  source matching source address
   *  @param[out] keys   list of the EventKey objects
   */
  void keys(std::list<EventKey>& keys, const Source& source) const
  {
    this->keysImpl(keys, source);
  }

protected:

  // Default constructor
  ProxyDictI () {}  
  
  
  /**
   *  @brief Add one more proxy object to the dictionary.
   *  
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] key     Event key for the data object.
   */
  virtual void putImpl( const boost::shared_ptr<ProxyI>& proxy, const EventKey& key ) = 0;

  /**
   *  @brief Get an object from event
   * 
   *  @param[in] typeinfo  Dynamic type info object
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @param[out] foundSrc If pointer is non-zero then pointed object will be assigned
   *                       with the exact source address of the returned object.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl( const std::type_info* typeinfo, 
                                           const Source& source, 
                                           const std::string& key,
                                           Pds::Src* foundSrc ) = 0;

  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  @param[in] key     Event key for the data object.
   *  @return true if proxy exists
   */
  virtual bool existsImpl(const EventKey& key) = 0;

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] key     Event key for the data object.
   *  @return false if object did not exist before this call
   */
  virtual bool removeImpl(const EventKey& key) = 0;

  /**
   *  @brief Get the list of event keys defined in event
   *  
   *  @param[in]  source matching source address
   *  @param[out] keys list of the EventKey objects
   */
  virtual void keysImpl(std::list<EventKey>& keys, const Source& source) const = 0;

private:


};

} // namespace PSEvt

#endif // PSEVT_PROXYDICTI_H
