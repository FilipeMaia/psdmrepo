#ifndef PSENV_CONFIGSTORE_H
#define PSENV_CONFIGSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigStore.
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
#include "PSEvt/ProxyDictI.h"
#include "PSEvt/DataProxy.h"
#include "PSEvt/EventKey.h"
#include "PSEvt/Source.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  @brief Class to store configuration object.
 *
 *  This class is very similar to PSEvt::Event class (and is implemented 
 *  on top of the same proxy dictionary stuff) but has limited interface.
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

class ConfigStore : boost::noncopyable {
public:

  // Special class used for type-less return from get()
  struct GetResultProxy {
    
    template<typename T>
    operator boost::shared_ptr<T>() {
      boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, std::string(), m_foundSrc);
      return boost::static_pointer_cast<T>(vptr);
    }
    
    boost::shared_ptr<PSEvt::ProxyDictI> m_dict;
    PSEvt::Source m_source;
    Pds::Src* m_foundSrc;
  };

  // Default constructor
  ConfigStore(const boost::shared_ptr<PSEvt::ProxyDictI>& dict) : m_dict(dict) {}

  // Destructor
  ~ConfigStore() {}

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
    const std::type_info& typeinfo = typeid(const T);
    if ( m_dict->exists(&typeinfo, source, std::string()) ) {
      m_dict->remove(&typeinfo, source, std::string());
    }
    m_dict->put(proxyPtr, &typeinfo, source, std::string());
  }
  
  /**
   *  @brief Get an object from store.
   *  
   *  @param[in] source Source detector address.
   *  @return Shared pointer which can be zero if object not found.
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
   *                       with the exact source address of the returned object (must 
   *                       be the same as source)
   *  @return Shared pointer which can be zero if object not found.
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
  std::list<PSEvt::EventKey> keys() const
  {
    std::list<PSEvt::EventKey> result;
    m_dict->keys(result);
    return result;
  }

protected:

private:

  // Data members
  boost::shared_ptr<PSEvt::ProxyDictI> m_dict;   ///< Proxy dictionary object 

};

} // namespace PSEnv

#endif // PSENV_CONFIGSTORE_H
