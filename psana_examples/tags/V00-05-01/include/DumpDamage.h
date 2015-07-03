#ifndef PSANA_EXAMPLES_DUMPDAMAGE_H
#define PSANA_EXAMPLES_DUMPDAMAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id:
//
// Description:
//	Class DumpDamage
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>

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
#include "PSEvt/DamageMap.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psana_examples {

/// @addtogroup psana_examples

/**
 *  @ingroup psana_examples
 *
 *  @brief Module class that dumps damage, event/config/calib keys, and changed config keys
 *
 *  This module is very similar to psana.EventKeys however it also dumps any damage
 *  information that is available as well as updates to the configStore. It reports
 *  on any keys that are found in more than one place.
 *
 *  Here is an example of some output:
 *
@verbatim
[info:DumpDamage]  event() run=0 calib=0 eventNumber=1 totalEvents= 1
    cfg --- --- ---            EventKey(type=Pds::Camera::FrameFexConfigV1, src=DetInfo(CxiDg4.0:Tm6740.0))
    cfg --- --- dmg 0x00000000 EventKey(type=Psana::Camera::FrameFexConfigV1, src=DetInfo(CxiDg4.0:Tm6740.0))
    --- --- evt ---            EventKey(type=PSEvt::EventId)
    --- --- evt ---            EventKey(type=Pds::Dgram)
    --- --- evt dmg 0x00000000 EventKey(type=Psana::Bld::BldDataEBeamV1, src=BldInfo(EBeam))
    --- --- --- dmg 0x00004000 EventKey(type=Psana::Ipimb::ConfigV1, src=BldInfo(NH2-SB1-IPM-01))
                    dropped=0 unititialized=0 OutOfOrder=0 OutOfSynch=0 UserDefined=1 IncompleteContribution=0 ContainsIncomplete=0 userBits=0x0
 -------- src damage with dropped contribs --------- 
   0x00000002  ProcInfo(ac.15.15.2b, pid=6ccb)
@endverbatim
 * Notice the counter which gives the event number within the calib cycle, as well as the
 * total number of events the module has seen.  
 * 
 * There are two sections of output, the list of keys and their damage, as well as src damage.
 * The first four columns indicate where the EventKey was found:
 * 
 * - cfg: in the ConfigStore
 * - clb: in the CalibStore
 * - evt: in the Event
 * - dmg: in the DamageMap
 *  
 * Keys that are in the damage map will also have their 32 bit damage value printed.
 * If this damage value is nonzero, then a second line is printed which indicates which
 * damage bits are on, and the value of the user damage bits.
 *
 * The second section, if present, means that src only damage was found. The damage, and then the
 * src are printed. A
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Psana::EventKeys PSEvt::DamageMap PSEvt::HistI
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

class DumpDamage : public Module {
public:

  // Default constructor
  DumpDamage (const std::string& name) ;

  // Destructor
  virtual ~DumpDamage () ;

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
  void printConfigKeyUpdates(Env &env);
  void printKeysAndDamage(std::ostream &out, Event &evt, Env &env);
private:
  long m_eventNumber, m_runNumber, m_calibCycleNumber;
  long m_totalEvents;
  std::map<PSEvt::EventKey, unsigned> m_configUpdates;
  std::map<PSEvt::EventKey, unsigned> m_calibUpdates;
  boost::shared_ptr<PSEvt::DamageMap> m_damageMap;
};

} // namespace psana_examples

#endif // PSANA_EXAMPLES_DUMPDAMAGE_H
