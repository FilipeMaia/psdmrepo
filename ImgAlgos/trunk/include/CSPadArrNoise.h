#ifndef IMGALGOS_CSPADARRNOISE_H
#define IMGALGOS_CSPADARRNOISE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrNoise.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

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

struct TwoIndexes {
  int i;
  int j;
};

struct MedianResult {
  double avg;
  double rms;
  double signal;
  double SoN;
};

class CSPadArrNoise : public CSPadBaseModule {
public:

  const static int NumColumns1 = NumColumns - 1;
  const static int NumRows1    = NumRows    - 1;
  
  // Default constructor
  CSPadArrNoise (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrNoise () ;

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

    virtual void procQuad(unsigned quad, const int16_t* data); 
    void collectStatInSect(unsigned quad, unsigned sect, const int16_t* sectData);
    MedianResult evaluateSoNForPixel(unsigned ic,unsigned ir,const int16_t* sectData);
    void evaluateVectorOfIndexesForMedian();
    void printMatrixOfIndexesForMedian();
    void printVectorOfIndexesForMedian();

    void printInputParameters();
    void printEventId(Event& evt);
    void printTimeStamp(Event& evt);

    void resetStatArrays();
  //void setCollectionMode();

    void procStatArrays();
    template <typename T>
    void saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows]);

private:

  std::string    m_fracFile;        // [out] file with fraction of noisy events in eac pixel
  std::string    m_maskFile;        // [out] file with mask 
  float          m_rmin;            // radial parameter of the area for median algorithm
  float          m_dr;              // radial band width of the area for median algorithm 
  float          m_SoNThr;
  float          m_frac_noisy_imgs;
  unsigned       m_print_bits;   

  unsigned       m_stat   [MaxQuads][MaxSectors][NumColumns][NumRows];
  uint16_t       m_mask   [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_status [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_signal [MaxQuads][MaxSectors][NumColumns][NumRows];

  std::vector<TwoIndexes> v_indForMediane;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRNOISE_H
