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
#include <typeinfo>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/DetInfo.hh"
#include "PsEvt/ProxyI.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsEvt {

/**
 *  @brief Interface for shared 
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
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  void put( const boost::shared_ptr<ProxyI>& proxy, 
            const std::type_info* typeinfo, 
            const Pds::DetInfo& detInfo, 
            const std::string& key ) 
  {
    this->putImpl(proxy, typeinfo, detInfo, key);
  }

  /**
   *  @brief Get an object from event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  boost::shared_ptr<void> get( const std::type_info* typeinfo, 
                               const Pds::DetInfo& detInfo, 
                               const std::string& key )
  {
   return this->getImpl(typeinfo, detInfo, key);
  }

  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  This is optimized version of get() which only checks whether the proxy
   *  is there but does not ask proxy to do any real work.
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if proxy exists
   */
  bool exists( const std::type_info* typeinfo, 
               const Pds::DetInfo& detInfo, 
               const std::string& key)
  {
   return this->existsImpl(typeinfo, detInfo, key);
  }

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  bool remove( const std::type_info* typeinfo, 
               const Pds::DetInfo& detInfo, 
               const std::string& key )
  {
   return this->removeImpl(typeinfo, detInfo, key);
  }

protected:

  // Default constructor
  ProxyDictI () {}  
  
  
  /**
   *  @brief Add one more proxy object to the dictionary.
   *  
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  virtual void putImpl( const boost::shared_ptr<ProxyI>& proxy, 
                        const std::type_info* typeinfo, 
                        const Pds::DetInfo& detInfo, 
                        const std::string& key ) = 0;

  /**
   *  @brief Get an object from event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl( const std::type_info* typeinfo, 
                                           const Pds::DetInfo& detInfo, 
                                           const std::string& key ) = 0;

  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if proxy exists
   */
  virtual bool existsImpl( const std::type_info* typeinfo, 
                           const Pds::DetInfo& detInfo, 
                           const std::string& key) = 0;

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  virtual bool removeImpl( const std::type_info* typeinfo, 
                           const Pds::DetInfo& detInfo, 
                           const std::string& key ) = 0;

private:


};

} // namespace PsEvt

#endif // PSEVT_PROXYDICTI_H
