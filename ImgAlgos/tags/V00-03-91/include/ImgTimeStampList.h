#ifndef IMGALGOS_IMGTIMESTAMPLIST_H
#define IMGALGOS_IMGTIMESTAMPLIST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgTimeStampList.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <fstream> // for std::ofstream operator << 
#include <sstream> // for stringstream 
//#include <typeinfo> // for typeid()

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

#include "PSEvt/Source.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief ImgTimeStampList gets the time stamps from events and save them in file. 
 *
 *  ImgTimeStampList psana module class is a simplified version of ImgVsTimeSplitInFiles.
 *  * saves timestamps for selected or all events in the text file
 *  * calculates sequential time index for each event.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImgTimeStampList : public Module {
public:

  //enum FILE_MODE {BINARY, TEXT};

  // Default constructor
  ImgTimeStampList (const std::string& name) ;

  // Destructor
  virtual ~ImgTimeStampList () ;

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

  void saveImageInFile(Event& evt);
  void printInputParameters();
  void printEventRecord(Event& evt, std::string comment=std::string());
  void printSummary(Event& evt, std::string comment=std::string());
  void printSummaryForParser(Event& evt, std::string comment=std::string());
  void openOutputFiles(Event& evt);
  void closeOutputFiles();
  void saveMetadataInFile();
  void saveTimeRecord(Event& evt);
  void evaluateMeanTimeBetweenEvents();
  void saveTimeRecordWithIndexInFile();

private:

  Pds::Src      m_src;
  Source        m_str_src;      // i.e. CxiDs1.0:Cspad.0
  std::string   m_key;          // i.e. Image2D
  std::string   m_fname;        // output file name for time stamp list
  unsigned      m_print_bits;
  long          m_count;

  std::string   m_fname_time;
  std::ofstream p_out_time;
  std::string   m_str_run_number;

  double        m_tsec_0;       // time of the 1st event
  double        m_tsec;         // time of current event
  double        m_tsec_prev;    // time of previous event
  double        m_dt;           // delta time between current and previous event
  unsigned      m_nevt;         // event number of current event frim eventID().vector()
  unsigned      m_nevt_prev;    // event number of previous event
  unsigned      m_sumt0;
  double        m_sumt1;
  double        m_sumt2;
  double        m_t_ave;        // average time between consecutive events in eventID().vector()
  double        m_t_rms;        // rms spread of dt 
  unsigned      m_tind_max;     // maximal value of the time index.

//protected:
//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGTIMESTAMPLIST_H
