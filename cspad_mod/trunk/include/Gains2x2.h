#ifndef CSPAD_MOD_GAINS2X2_H
#define CSPAD_MOD_GAINS2X2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gains2x2.
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
#include "psddl_psana/cspad.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/// @addtogroup cspad_mod

/**
 *  @ingroup cspad_mod
 *
 *  @brief Example module class for psana
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Philip Hart
 */

class Gains2x2 : public Module {
public:
    
  enum { MAXQUADS = Psana::CsPad::MaxQuadsPerSensor };
  enum { MAXTWOXONES = Psana::CsPad::SectorsPerQuad };
  enum { COLS = Psana::CsPad::ColumnsPerASIC };
  enum { ROWS = Psana::CsPad::MaxRowsPerASIC*2 };


  // Default constructor
  Gains2x2 (const std::string& name) ;

  // Destructor
  virtual ~Gains2x2 () ;

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

private:

  // Data members, this is for example purposes only
  
  Source m_src;         // Data source set from config file
  unsigned m_maxEvents;
  bool m_filter;
  long m_count;
  std::string m_calibKey;
  float m_minClusterE;
  float m_maxSingleNeighborE;
  std::string m_gainFile;
  std::string m_rootFile;

  TH1D *pixelPeakSub[MAXQUADS][MAXTWOXONES];
  TH1D *pixelPeakSubSingles[MAXQUADS][MAXTWOXONES];

  TH1I *singlePhotonPeak[MAXQUADS][MAXTWOXONES][COLS][ROWS];
  TH1I *singlePixelCluster[MAXQUADS][MAXTWOXONES][COLS][ROWS];
  TH1I *doublePixelCluster[MAXQUADS][MAXTWOXONES][COLS][ROWS];
  TH1I *clusterE[MAXQUADS][MAXTWOXONES][COLS][ROWS];

  unsigned _quads, _twoXones;

};

} // namespace cspad_mod

#endif // CSPAD_MOD_GAINS2X2_H
