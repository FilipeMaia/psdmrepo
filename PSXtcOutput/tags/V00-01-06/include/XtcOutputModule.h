#ifndef PSXTCOUTPUT_XTCOUTPUTMODULE_H
#define PSXTCOUTPUT_XTCOUTPUTMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcOutputModule.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Dgram.h"
#include "XtcInput/XtcFilter.h"
#include "XtcInput/XtcFilterTypeId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcOutput {

/// @addtogroup PSXtcOutput

/**
 *  @ingroup PSXtcOutput
 *
 *  @brief Class which writes XTC datagrams to a file(s)
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class XtcOutputModule : public Module {
public:

  // Default constructor
  XtcOutputModule (const std::string& name) ;

  // Destructor
  virtual ~XtcOutputModule () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

  /// Method that writes the data to output file, opening and closing it if necessary
  void saveData(Pds::Dgram* dg, Pds::TransitionId::Value transition);

private:

  unsigned m_chunkSizeMB;   ///< Maximum chunk size in MB
  std::string m_nameFmt;    ///< Format as for boost::format, %1 is expNum, %2 is run, %3 is stream, %4 is chunk
  std::string m_dirName;    ///< Directory where to store output
  bool m_keepEpics;         ///< Keep EPICS data in datagrams which are otherwise discarded
  int m_expNum;             ///< Experiment number
  int m_run;                ///< Run number
  int m_stream;             ///< Stream number
  int m_chunk;              ///< Current chunk number
  int m_fd;                 ///< File descriptor for output file
  size_t m_storedBytes;     ///< Total stored bytes in current chunk
  boost::shared_ptr<XtcInput::XtcFilter<XtcInput::XtcFilterTypeId> > m_filter;   ///< Filter for EPICS
};

} // namespace PSXtcOutput

#endif // PSXTCOUTPUT_XTCOUTPUTMODULE_H
