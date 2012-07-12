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
 *  @brief Implementation of the EventId interface for XTC events.
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

class XtcEventId : public PSEvt::EventId {
public:

  // Default constructor
  XtcEventId (int run, const PSTime::Time& time) ;

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

};

} // namespace PSXtcInput

#endif // PSXTCINPUT_XTCEVENTID_H
