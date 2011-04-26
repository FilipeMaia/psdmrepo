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
#include "PSEvt/ProxyDictI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventKey.h"
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

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
   *  @param[in] typeinfo  Dynamic type info object
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   */
  virtual void putImpl( const boost::shared_ptr<ProxyI>& proxy, 
                        const std::type_info* typeinfo, 
                        const Pds::Src& source, 
                        const std::string& key ) ;

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
                                           Pds::Src* foundSrc );

  /**
   *  @brief Check if proxy of given type exists in the event
   *  
   *  @param[in] typeinfo  Dynamic type info object
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return true if proxy exists
   */
  virtual bool existsImpl( const std::type_info* typeinfo, 
                           const Pds::Src& source, 
                           const std::string& key) ;

  /**
   *  @brief Remove object of given type from the event
   *  
   *  @param[in] typeinfo  Dynamic type info object
   *  @param[in] source Source detector address.
   *  @param[in] key     Optional key to distinguish different objects of the same type.
   *  @return false if object did not exist before this call
   */
  virtual bool removeImpl( const std::type_info* typeinfo, 
                           const Pds::Src& source, 
                           const std::string& key ) ;

  /**
   *  @brief Get the list of event keys defined in event
   *  
   *  @param[out] keys list of the EventKey objects
   */
  virtual void keysImpl(std::list<EventKey>& keys) const;

private:

  typedef boost::shared_ptr<ProxyI> proxy_ptr;
  typedef std::map<EventKey,proxy_ptr> Dict;

  // Data members
  Dict m_dict;

};

} // namespace PSEvt

#endif // PSEVT_PROXYDICT_H
