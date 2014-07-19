#ifndef PSXTCMPINPUT_DGRAMSOURCEWORKER_H
#define PSXTCMPINPUT_DGRAMSOURCEWORKER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramSourceWorker.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "PSXtcInput/IDatagramSource.h"
#include "psana/Configurable.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcMPInput {

/// @addtogroup PSXtcMPInput

/**
 *  @ingroup PSXtcMPInput
 *
 *  @brief Implementation of the IDatagramSource which reads datagrams
 *  from master process, to be used in worker process.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DgramSourceWorker : public PSXtcInput::IDatagramSource, public psana::Configurable {
public:

  // Constructor tagkes the name of the input module, used for accessing module
  // configuration parameters.
  DgramSourceWorker(const std::string& name);

  // Destructor
  virtual ~DgramSourceWorker () ;

  /**
   *   Initialization method for datagram source, this is typically called
   *   in beginJob() method and it may contain initialization code which
   *   cannot be executed during construction of an instance.
   */
  virtual void init();

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
  virtual bool next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg);

protected:

private:

  int m_fdDataPipe;   ///< file descriptor for data pipe
  int m_workerId;     ///< worker process number
  int m_fdReadyPipe;  ///< fd of the ready pipe
  bool m_ready;       ///< means that ready flag was sent and not consumed by master

};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_DGRAMSOURCEWORKER_H
