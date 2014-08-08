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
 *  This example demonstrates how to deal with the raw xtc dgram, if it is present in
 *  the event (it will not be if psana's input is hdf5). Ealier versions of psana put 
 *  a single Pds::Dgram into the event. Later versions that deal with offline event 
 *  building put a XtcInput::DgramList into the event. 
 *
 *  In general users *should not* work with this data. This is the raw data from which psana 
 *  fills the Event. Users should access the data by getting psana Types from the Event. In general
 *  they should not be looking at the Datagrams. However looking at the Datagrams could be 
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
