#ifndef IMGALGOS_CSPADARRPEAKANALYSIS_H
#define IMGALGOS_CSPADARRPEAKANALYSIS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrPeakAnalysis.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/CSPadArrPeakFinder.h"

//#include "root/TFile.h"
//#include "root/TH1D.h"
//#include "root/TTree.h"
//#include "root/TBranch.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CSPadArrPeakAnalysis : public Module {
public:

  // Default constructor
  CSPadArrPeakAnalysis (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrPeakAnalysis () ;

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
  void procEvent(Event& evt);
  //void collectStatistics();
  //void initStatistics();
  //void summStatistics();
  void printPeaks();
  void printInputParameters();
  void printEventId(Event& evt);
  void printTimeStamp(Event& evt);

private:
  Pds::Src    m_src;
  Source      m_str_src;
  std::string m_key;
  //std::string m_fname_root;
  unsigned m_print_bits;
  long     m_count;
  long     m_selected;

  std::vector<Peak>* m_peaks;

  Peak     m_peak;

  //TFile*   m_tfile;
  //TH1D*    m_his01;
  //TTree*   m_tree;
  //TBranch* m_branch;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRPEAKANALYSIS_H
