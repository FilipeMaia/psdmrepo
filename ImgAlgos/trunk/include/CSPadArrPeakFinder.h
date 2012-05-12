#ifndef IMGALGOS_CSPADARRPEAKFINDER_H
#define IMGALGOS_CSPADARRPEAKFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrPeakFinder.
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
#include "ImgAlgos/CSPadMaskV1.h"

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
  double sig;
  double SoN;
};

struct Peak{
   double quad;
   double sect; 
   double col;
   double row; 
   double sigma_col;
   double sigma_row;
   double ampmax;
   double amptot;
   unsigned npix;
};


class CSPadArrPeakFinder : public Module {
public:

    enum { MaxQuads      = Psana::CsPad::MaxQuadsPerSensor }; // 4
    enum { MaxSectors    = Psana::CsPad::SectorsPerQuad    }; // 8
    enum { NumColumns    = Psana::CsPad::ColumnsPerASIC    }; // 185 THERE IS A MESS IN ONLINE COLS<->ROWS
    enum { NumRows       = Psana::CsPad::MaxRowsPerASIC*2  }; // 388 THERE IS A MESS IN ONLINE COLS<->ROWS 
    enum { SectorSize    = NumColumns * NumRows            }; // 185 * 388
    enum { SIZE_OF_ARRAY = MaxQuads * MaxSectors * SectorSize }; 
    enum { NumColumns1   = NumColumns - 1};
    enum { NumRows1      = NumRows    - 1};
    enum SELECTION_MODE{ SELECTION_OFF, SELECTION_ON, SELECTION_INV };
  
  // Default constructor
  CSPadArrPeakFinder (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrPeakFinder () ;

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
    void findPeaksInSect(unsigned quad, unsigned sect);
    MedianResult evaluateSoNForPixel(unsigned ic,unsigned ir,const int16_t* sectData);
    void iterateOverConnectedPixels(int ic, int ir);
    bool peakSelector();
    void savePeakInfo();
    void printVectorOfPeaks();
    void maskUpdateControl();
    void setSelectionMode();
    bool eventSelector();

    void evaluateVectorOfIndexesForMedian();
    void printMatrixOfIndexesForMedian();
    void printVectorOfIndexesForMedian();

    void printInputParameters();
    void printEventId(Event& evt);
    void printTimeStamp(Event& evt);

    void resetForEventProcessing();
    void resetMaskOfNoisyPix();
    void resetStatArrays();
    void resetSignalArrays();

    void procStatArrays();
    template <typename T>
    void saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows]);

    void getMaskFromFile();
    void printMaskStatistics();

    std::string strEventCounter();
    std::string strTimeStamp(Event& evt);
    std::string strRunNumber(Event& evt);
    void doOperationsForSelectedEvents(Event& evt);
    void savePeaksInEvent(Event& evt);

private:
  //Source         m_src;             // Data source set from config file
  Pds::Src       m_src;             // source address of the data object
  std::string    m_str_src;         // string with source name
  std::string    m_key;             // string with key name
  std::string    m_key_peaks_out;   // string with key for found peaks in selected events
  std::string    m_maskFile_inp;    // [in]  file with mask 
  std::string    m_maskFile_out;    // [out] file with mask 
  std::string    m_fracFile_out;    // [out] file with pixel status info: fraction of noisy images (events)
  std::string    m_evtFile_out;     // [out] file name prefix for event array output
  float          m_rmin;            // radial parameter of the area for median algorithm
  float          m_dr;              // radial band width of the area for median algorithm 
  float          m_SoNThr;          // S/N threshold for outlier pix finder
  float          m_frac_noisy_imgs; // For hot-pixel mask definition

  unsigned       m_peak_npix_min;    // Peak selection parameters
  unsigned       m_peak_npix_max;    // 
  double         m_peak_amp_tot_thr; //

  unsigned       m_event_npeak_min;  // Minimum number of peaks in the event for selector
  double         m_event_amp_tot_thr;// Amplitude threshold on total signal amplitude in all peaks

  unsigned       m_nevents_mask_update;
  unsigned       m_nevents_mask_accum;

  std::string    m_sel_mode_str;
  SELECTION_MODE m_sel_mode;
  unsigned       m_out_file_bits;     // Yes/No bits to control output files
  unsigned       m_print_bits;   
  unsigned long  m_count;             // number of events from the beginning of job
  unsigned long  m_count_mask_update; // number of events from the beginning of job
  unsigned long  m_count_mask_accum;  // number of events from the beginning of job

  unsigned       m_segMask         [MaxQuads];  // segment masks per quadrant
  unsigned       m_stat            [MaxQuads][MaxSectors][NumColumns][NumRows];
  int16_t        m_mask            [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_frac_noisy_evts [MaxQuads][MaxSectors][NumColumns][NumRows];
  int16_t        m_signal          [MaxQuads][MaxSectors][NumColumns][NumRows];
  uint16_t       m_proc_status     [MaxQuads][MaxSectors][NumColumns][NumRows];

  ImgAlgos::CSPadMaskV1 *m_mask_initial;

  std::vector<TwoIndexes> v_indForMediane;

  // Stuff for the peak finding algorithm
  unsigned  m_peak_npix;
  double    m_peak_amp_tot;
  double    m_peak_amp_max;
  double    m_peak_amp_x_col1;
  double    m_peak_amp_x_col2;
  double    m_peak_amp_x_row1;
  double    m_peak_amp_x_row2;

  std::vector<Peak> v_peaks;
  unsigned  m_quad;
  unsigned  m_sect;
  double    m_event_amp_tot;

};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRPEAKFINDER_H
