#ifndef PSXTCINPUT_IDATAGRAMSOURCE_H
#define PSXTCINPUT_IDATAGRAMSOURCE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IDatagramSource.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/// @addtogroup PSXtcInput

/**
 *  @ingroup PSXtcInput
 *
 *  @brief Declaration of interface for datagram source classes.
 *
 *  Datagram source is an abstraction of a  sequence of datagrams that can be iterated 
 *  over. Instance method next() is called repeatedly to obtain next available datagram
 *  until this method returns empty instance. 
 *  
 *  In the future this interface could be extended to support direct access based on 
 *  some index (e.g. time-based).
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class IDatagramSource {
public:

  // Destructor
  virtual ~IDatagramSource() {}

  /**
   *   Initialization method for datagram source, this is typically called
   *   in beginJob() method and it may contain initialization code which 
   *   cannot be executed during construction of an instance.
   */ 
  virtual void init() = 0;

  /**
   *  @brief Return next datagram(s) from the source.
   *
   *  This method returns two sets of datagrams - eventDg is the set of histograms
   *  belonging to the next event, nonEventDg is the set of datagrams which has some
   *  other data (like EPICS) which is needed for correct interpretation of current
   *  event. Currently eventDg should contain one datagram but potentially in the
   *  future we may start event building in offline and that list can grow longer.
   *  It nonEventDg is non-empty then it has to be processed first as those datagram
   *  should come from earlier time than eventDg and eventDg may contain data that
   *  overrides data in nonEventDg (e.g. some EPICS PV data may be contained in both
   *  nonEventDg and eventDg).
   *
   *  This method will called repeatedly until it returns false.
   *
   *  @param[out] eventDg    returned set of datagrams for current event
   *  @param[out] nonEventDg returned set of datagrams containing other information.
   *  @return false if there are no more events, both eventDg and nonEventDg will be empty in this case.
   */
  virtual bool next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg) = 0;

protected:

  // Default constructor
  IDatagramSource()  {}

private:

  // disable copy
  IDatagramSource(const IDatagramSource&);
  IDatagramSource& operator=(const IDatagramSource&);
  
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_IDATAGRAMSOURCE_H
