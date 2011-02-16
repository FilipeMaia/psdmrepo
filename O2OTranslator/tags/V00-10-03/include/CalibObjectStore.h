#ifndef O2OTRANSLATOR_CALIBOBJECTSTORE_H
#define O2OTRANSLATOR_CALIBOBJECTSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibObjectStore.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

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

namespace O2OTranslator {

/**
 *  Storage for calibration objects.
 *  
 *  Object in the calibration store are identified by their C++ type
 *  and address of the detector.
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

class CalibObjectStore : boost::noncopyable {
public:
  
  // Default constructor
  CalibObjectStore () ;

  // Destructor
  ~CalibObjectStore () ;

  // Add one more object to the store
  template <typename Data>
  void add(boost::shared_ptr<Data> data, const Pds::DetInfo& address);
  
  // Get object from store
  template <typename Data>
  boost::shared_ptr<Data> get(const Pds::DetInfo& address) const;
  
  
protected:

private:
  
  // uniquely identifying key for stored object
  struct Key {
    Key(const std::type_info* a_typeinfo, const Pds::DetInfo& a_address) 
      : typeinfo(a_typeinfo), phy(a_address.phy()) {}
    bool operator<(const Key& other) const {
      if (phy < other.phy) return true;
      if (phy > other.phy) return false;
      if( typeinfo->before(*other.typeinfo) ) return true;
      return false;
    }
    const std::type_info* typeinfo;
    const uint32_t phy;
  };
  
  // void pointer ash boost shared pointer
  typedef boost::shared_ptr<void> void_ptr;

  typedef std::map<Key,void_ptr> ObjectStore;
  
  // Data members
  ObjectStore m_store;

};



// Add one more object to the store
template <typename Data>
void 
CalibObjectStore::add(boost::shared_ptr<Data> data, const Pds::DetInfo& address)
{
  // do not store empty objects
  if ( not data.get() ) return;

  // store is as void, type will be recovered later
  void_ptr vptr = boost::static_pointer_cast<void>(data);
  Key key(&typeid(Data), address);
  m_store[key] = vptr;
}

// Get object from store
template <typename Data>
boost::shared_ptr<Data> 
CalibObjectStore::get(const Pds::DetInfo& address) const
{
  Key key(&typeid(Data), address);
  ObjectStore::const_iterator it = m_store.find(key);
  if (it == m_store.end()) return boost::shared_ptr<Data>();
  
  return boost::static_pointer_cast<Data>(it->second);
}

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CALIBOBJECTSTORE_H
