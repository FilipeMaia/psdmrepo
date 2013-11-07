#ifndef PSXTCINPUT_XTCEVENTID_H
#define PSXTCINPUT_XTCEVENTID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcEventId.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "PSEvt/EventId.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/**
 *  @ingroup PSXtcInput
 *
 *  @brief Implementation of the EventId interface for XTC events.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcEventId : public PSEvt::EventId {
public:

  // Default constructor
  XtcEventId (int run, const PSTime::Time& time, unsigned fiducials, unsigned ticks, unsigned vector, unsigned control) ;

  // Destructor
  ~XtcEventId () ;

  /**
   *  @brief Return the time for event.
   */
  virtual PSTime::Time time() const;

  /**
   *  @brief Return the run number for event.
   *
   *  If run number is not known -1 will be returned.
   */
  virtual int run() const;

  /**
   *  @brief Returns fiducials counter for the event.
   *
   *  Note that MCC sends fiducials as 17-bit number which overflows
   *  frequently (fiducials clock runs at 360Hz) so this number is
   *  not unique. In some cases (e.g. when reading from old HDF5
   *  files) fiducials is not know, 0 will be returned in this case.
   */
  virtual unsigned fiducials() const;

  /**
   *  @brief Returns 119MHz counter within the fiducial.
   *
   *  Returns the value of 119MHz counter within the fiducial for the
   *  event code which initiated the readout. In some cases (e.g. when
   *  reading from old HDF5 files) ticks are not know, 0 will be
   *  returned in this case.
   */
  virtual unsigned ticks() const;

  /**
   *  @brief Returns event counter since Configure.
   *
   *  Note that counter is saved as 15-bits integer and will overflow
   *  frequently. In some cases (e.g. when reading from old HDF5
   *  files) counter is not know, 0 will be returned  in this case.
   */
  virtual unsigned vector() const;

  virtual unsigned control() const;

  /// check if two event IDs refer to the same event
  virtual bool operator==(const EventId& other) const;
  
  /// Compare two event IDs for ordering purpose
  virtual bool operator<(const EventId& other) const;
  
  /// Dump object in human-readable format
  virtual void print(std::ostream& os) const;
  
protected:

private:

  // Data members
  int m_run;
  PSTime::Time m_time;
  unsigned m_fiducials;
  unsigned m_ticks;
  unsigned m_vector;
  unsigned m_control;
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCEVENTID_H
