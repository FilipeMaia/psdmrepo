#ifndef PSEVT_HISTI_H
#define PSEVT_HISTI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: 
//
// Description:
//	Class HistI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventKey.h"

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
 *  @brief Class defining a history interface.
 *  
 *  This class supports a simple history interface, total number of
 *  updates, and total number of updates for a specific EventKey.  
 *
 *  @see ProxyDictI
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

class HistI : boost::noncopyable {
public:

  // Destructor
  virtual ~HistI () {}

  /**
   *  @brief implement to provide number of updates
   *
   * Derived classes can provide a simple history mechanism by 
   * overriding this class and returning the number of updates
   *  
   *  @param[out] total number of updates
   */
  virtual long totalUpdates() const = 0;

  /**
   *  @brief implement to provide total number of updates for a key.
   *
   *  Derived classes can implement this to provide a simple history on a key - 
   *  the total number of updates
   *  
   *  @param[in]  key EventKey to get updates for
   *  @param[out] number of updates
   */
  virtual long updates(const EventKey &) const = 0;
};

} // namespace PSEvt

#endif // PSEVT_HISTI_H
