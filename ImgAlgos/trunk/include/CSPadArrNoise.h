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
#include "psana/Module.h"

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

class CSPadArrNoise : public Module {
public:

    enum { MaxQuads   = Psana::CsPad::MaxQuadsPerSensor }; // 4
    enum { MaxSectors = Psana::CsPad::SectorsPerQuad    }; // 8
    enum { NumColumns = Psana::CsPad::ColumnsPerASIC    }; // 185 THERE IS A MESS IN ONLINE COLS<->ROWS
    enum { NumRows    = Psana::CsPad::MaxRowsPerASIC*2  }; // 388 THERE IS A MESS IN ONLINE COLS<->ROWS 
    enum { SectorSize = NumColumns * NumRows            }; // 185 * 388
    enum { NumColumns1= NumColumns - 1};
    enum { NumRows1   = NumRows    - 1};
  
  // Default constructor
  CSPadArrNoise (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrNoise () ;

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
    void collectStatInQuad(unsigned quad, const int16_t* data);
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
  //Source         m_src;             // Data source set from config file
  Pds::Src       m_src;             // source address of the data object
  std::string    m_str_src;         // string with source name
  std::string    m_key;             // string with key name
  std::string    m_statusFile;      // [out] file with pixel status info: fraction of noisy images (events)
  std::string    m_maskFile;        // [out] file with mask 
  float          m_rmin;            // radial parameter of the area for median algorithm
  float          m_dr;              // radial band width of the area for median algorithm 
  float          m_SoNThr;
  float          m_frac_noisy_imgs;
  unsigned       m_print_bits;   
  unsigned long  m_count;  // number of events from the beginning of job

  unsigned       m_segMask[MaxQuads];  // segment masks per quadrant
  unsigned       m_stat   [MaxQuads][MaxSectors][NumColumns][NumRows];
  uint16_t       m_mask   [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_status [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_signal [MaxQuads][MaxSectors][NumColumns][NumRows];

  std::vector<TwoIndexes> v_indForMediane;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRNOISE_H
