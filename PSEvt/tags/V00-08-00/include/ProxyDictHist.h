#ifndef PSEVT_PROXYDICTHIST_H
#define PSEVT_PROXYDICTHIST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: ProxyDict.h 2579 2011-10-28 22:11:09Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//	Class ProxyDictHist.  keeps track of the number of updates done to each key.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/ProxyDict.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
 *  @brief A ProxyDict that implements the HistI interface.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

class ProxyDictHist : public ProxyDict, private HistI {
public:

  ProxyDictHist(const boost::shared_ptr<AliasMap>& amap) : ProxyDict(amap), m_totalUpdates(0) {}
 
 /**
   *  @brief returns total number of put and remove calls made with dictionary
   */
  virtual long totalUpdates() const { return m_totalUpdates; }

  /**
   *  @brief returns total number of put and remove calls made for given key
   *  
   *  @param[in] key     key to lookup update count for
   */
  virtual long updates(const EventKey &key) const { return m_updates[key]; }

  virtual const HistI * hist() const { return this; }

protected:

  /**
   *  @brief updates count after adding using base class to add to dict.
   *  
   *  @param[in] proxy   Proxy object for type T.
   *  @param[in] key     Event key for the data object.
   */
  virtual void putImpl( const boost::shared_ptr<ProxyI>& proxy, const EventKey& key ) {
    ProxyDict::putImpl(proxy, key);
    ++m_updates[key];
    ++m_totalUpdates;
  }

  /**
   *  @brief updates count after adding using base class to remove from to dict.
   *  
   *  Does not update count if key was not present. 
   *
   *  @param[in] key     Event key for the data object.
   *  @return false if object did not exist before this call
   */
  virtual bool removeImpl( const EventKey& key ) {
    bool objectExistedBeforeCall = ProxyDict::removeImpl(key);
    if (objectExistedBeforeCall) {
      ++m_updates[key];
      ++m_totalUpdates;
    }
    return objectExistedBeforeCall;
  }
private:
  long m_totalUpdates;
  mutable std::map<EventKey, long> m_updates;
};

} // namespace PSEvt

#endif // PSEVT_PROXYDICTPUTHIST_H
