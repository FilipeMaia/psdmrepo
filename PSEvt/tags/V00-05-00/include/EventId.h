#ifndef PSEVT_EVENTID_H
#define PSEVT_EVENTID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EventId.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iosfwd>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSTime {
  class Time;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Class defining abstract interface for Event ID objects.
 *  
 *  Event ID should include enough information to uniquely identify
 *  an event (and possibly a location of the event in data file).
 *  Currently we include event timestamp (PSTime::Time object) and
 *  run number into Event ID.
 *  
 *  Implementation of this interface will probably be tied to a 
 *  particular input data format so the interface will be implemented 
 *  in the packages responsible for reading data (e.g. PSXtcInput). 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class EventId : boost::noncopyable {
public:

  // Destructor
  virtual ~EventId () {}

  /**
   *  @brief Return the time for event.
   */
  virtual PSTime::Time time() const = 0;

  /**
   *  @brief Return the run number for event.
   *  
   *  If run number is not known -1 will be returned.
   */
  virtual int run() const = 0;
  
  /**
   *  @brief Returns fiducials counter for the event.
   *
   *  Note that MCC sends fiducials as 17-bit number which overflows
   *  frequently (fiducials clock runs at 360Hz) so this number is
   *  not unique. In some cases (e.g. when reading from old HDF5
   *  files) fiducials is not know, 0 will be returned in this case.
   */
  virtual unsigned fiducials() const = 0;

  /**
   *  @brief Returns event counter since Configure.
   *
   *  Note that counter is saved as 15-bits integer and will overflow
   *  frequently. In some cases (e.g. when reading from old HDF5
   *  files) counter is not know, 0 will be returned  in this case.
   */
  virtual unsigned vector() const = 0;

  /// check if two event IDs refer to the same event
  virtual bool operator==(const EventId& other) const = 0;
  
  /// Compare two event IDs for ordering purpose
  virtual bool operator<(const EventId& other) const = 0;
  
  /// Dump object in human-readable format
  virtual void print(std::ostream& os) const = 0;
  
protected:

  // Default constructor
  EventId () {}

private:

};

/// Standard stream insertion operator
std::ostream&
operator<<(std::ostream& os, const EventId& eid);

} // namespace PSEvt

#endif // PSEVT_EVENTID_H
