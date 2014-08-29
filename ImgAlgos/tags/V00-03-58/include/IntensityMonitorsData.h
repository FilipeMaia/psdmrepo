#ifndef IMGALGOS_INTENSITYMONITORSDATA_H
#define IMGALGOS_INTENSITYMONITORSDATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IntensityMonitorsData.
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
 *  @brief IntensityMonitorsData gets the data from a few intensity monitors and save them in file.. 
 *
 *  IntensityMonitorsData psana module class is a simplified version of ImgVsTimeSplitInFiles.
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

struct Quartet{
  float v1;
  float v2; 
  float v3;
  float v4; 
   
  Quartet(float p1, float p2, float p3, float p4): v1(p1), v2(p2), v3(p3), v4(p4) {}
};

 
class IntensityMonitorsData : public Module {
public:

  enum FILE_MODE {BINARY, TEXT};

  // Default constructor
  IntensityMonitorsData (const std::string& name) ;

  // Destructor
  virtual ~IntensityMonitorsData () ;

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

  void setFileMode();
  void printInputParameters();
  void printEventRecord(Event& evt, std::string comment=std::string());
  void printSummary(Event& evt, std::string comment=std::string());
  void printSummaryForParser(Event& evt, std::string comment=std::string());
  void openOutputFiles();
  void closeOutputFiles();
  //void makeListOfSources(); 
  void procEvent(Event& evt, Env& env);
  std::string strOfSources();
  std::string strRecord(Event& evt, Env& env);
  float*      arrRecord(Event& evt, Env& env);
  void printDataForSource (Event& evt, Env& env, Source& src);
  Quartet getDataForSource(Event& evt, Env& env, Source& src);


private:
  //Pds::Src      m_src;
  Source*       m_src_list;
  int           m_size_of_list;
  int           m_size_of_arr;
  std::string   m_file_type;    // file type "txt" or "bin" 
  FILE_MODE     m_file_mode; 
  std::string   m_fname;        // output file name for intensity monitors' data
  std::string   m_fname_header; // output file name for comments in stead of header
  unsigned      m_print_bits;
  long          m_count;

  std::string   m_str_run_number;
  std::ofstream p_out;
  std::ofstream p_out_header;

//protected:
//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_INTENSITYMONITORSDATA_H
