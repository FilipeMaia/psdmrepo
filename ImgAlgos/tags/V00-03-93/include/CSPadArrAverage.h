#ifndef IMGALGOS_CSPADARRAVERAGE_H
#define IMGALGOS_CSPADARRAVERAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrAverage.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "ImgAlgos/CSPadBaseModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"

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

class CSPadArrAverage : public CSPadBaseModule {
public:

  // Default constructor
  CSPadArrAverage (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrAverage () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
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
  virtual void initData();
  virtual void procQuad(unsigned quad, const int16_t* data);
  void printInputParameters();
  void printEventId(Event& evt);
  void resetStatArrays();
  void procStatArrays();
  void saveCSPadArrayInFile(std::string& fname, double arr[MaxQuads][MaxSectors][NumColumns][NumRows]);

private:
  std::string    m_aveFile;
  std::string    m_rmsFile;
  unsigned       m_print_bits;   
  unsigned long  m_nev_stage1;
  unsigned long  m_nev_stage2;
  double         m_gate_width1;
  double         m_gate_width2;

  double         m_gate_width;
  unsigned       m_stat[MaxQuads][MaxSectors][NumColumns][NumRows];  // statistics per pixel
  double         m_sum [MaxQuads][MaxSectors][NumColumns][NumRows];  // sum per pixel
  double         m_sum2[MaxQuads][MaxSectors][NumColumns][NumRows];  // sum of squares per pixel
  double         m_ave [MaxQuads][MaxSectors][NumColumns][NumRows];  // average per pixel
  double         m_rms [MaxQuads][MaxSectors][NumColumns][NumRows];  // rms per pixel
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRAVERAGE_H
