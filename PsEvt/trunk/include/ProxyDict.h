#ifndef PSEVT_PROXYDICT_H
#define PSEVT_PROXYDICT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProxyDict.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>
#include <typeinfo>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "PsEvt/ProxyDictI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/DetInfo.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsEvt {

/**
 *  @brief Implementation of the proxy dictionary interface.
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

class ProxyDict : public ProxyDictI {
public:

  // Default constructor
  ProxyDict () ;

  // Destructor
  virtual ~ProxyDict () ;

protected:

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
                        const std::string& key ) ;

  /**
   *  @brief Get an object from event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return Shared pointer of void type.
   */
  virtual boost::shared_ptr<void> getImpl( const std::type_info* typeinfo, 
                                           const Pds::DetInfo& detInfo, 
                                           const std::string& key ) ;

  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if proxy exists
   */
  virtual bool existsImpl( const std::type_info* typeinfo, 
                           const Pds::DetInfo& detInfo, 
                           const std::string& key) ;

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] detInfo Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  virtual bool removeImpl( const std::type_info* typeinfo, 
                           const Pds::DetInfo& detInfo, 
                           const std::string& key ) ;

private:

  // uniquely identifying key for stored object
  struct Key {
    Key(const std::type_info* a_typeinfo, const Pds::DetInfo& a_address, const std::string& a_key) 
      : typeinfo(a_typeinfo), phy(a_address.phy()), key(a_key) {}
    
    bool operator<(const Key& other) const 
    {
      if (phy < other.phy) return true;
      if (phy > other.phy) return false;
      if( typeinfo->before(*other.typeinfo) ) return true;
      if( other.typeinfo->before(*typeinfo) ) return false;
      if (key < other.key) return true;
      return false;
    }
    
    const std::type_info* typeinfo;
    const uint32_t phy;
    const std::string key;
  };

  typedef boost::shared_ptr<ProxyI> proxy_ptr;
  typedef std::map<Key,proxy_ptr> Dict;

  // Data members
  Dict m_dict;

};

} // namespace PsEvt

#endif // PSEVT_PROXYDICT_H
