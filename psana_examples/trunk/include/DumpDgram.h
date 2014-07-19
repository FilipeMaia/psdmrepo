#ifndef PSANA_EXAMPLES_DUMPDGRAM_H
#define PSANA_EXAMPLES_DUMPDGRAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id:
//
// Description:
//	Class DumpDgram
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/DgramList.h"
#include "XtcInput/FiducialsCompare.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psana_examples {

/// @addtogroup psana_examples

/**
 *  @ingroup psana_examples
 *
 *  @brief gets the DgramList or Dgram from the Event. Dumps basic information.
 *
 *  Prior to merging control streams, the Event would contain a shared pointer to the raw data
 *  from which the Event was formed. The raw data type is Pds::Dgram. After adding features to psana to merge
 *  the control streams with the DAQ streams, an event may have been formed from two or more pieces 
 *  of raw data rather than just one. At this point, psana no longer puts a Pds::Dgram in the 
 *  Event, rather it puts a XtcInput::DgramList into the Event.
 *
 *  This example demonstrates how to deal with both - the XtcInput::DgramList and the Pds::Dgram. 
 *  In general users *should not* work with this data. This is the raw data from which psana 
 *  fills the Event. Users should access the data by getting psana Types from the Event. They 
 *  should not, in general be looking at the Datagrams. However looking at the Datagrams could be 
 *  useful for diagnosing problems with event building.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see XtcInput::DgramList
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

class DumpDgram : public Module {
public:

  // Default constructor
  DumpDgram (const std::string& name) ;

  // Destructor
  virtual ~DumpDgram () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);

  /// Method which is called with event data
  virtual void event(Event& evt, Env& env);

  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

 protected:
  void dgramDump(Event &evt, const std::string &hdr);
};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DGRAMLISTFIDUCIALS_H
