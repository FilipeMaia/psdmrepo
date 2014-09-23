#ifndef IMGALGOS_IMGPEAKFINDERAB_H
#define IMGALGOS_IMGPEAKFINDERAB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgPeakFinderAB.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
//#include <omp.h>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/ImgParametersV1.h"
#include "ImgAlgos/TimeInterval.h"

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
  double row; 
  double col;
  double sigma_row;
  double sigma_col;
  double ampmax;
  double amptot;
  double bkgdtot;
  double noise;
  double SoN;
  unsigned npix;
};

// Stuff for the peak finding algorithm in thread-safe mode
struct PeakWork{
  unsigned  peak_npix;
  double    peak_SoN;
  double    peak_bkgd_tot;
  double    peak_noise2_tot;
  double    peak_noise;
  double    peak_amp_tot;
  double    peak_amp_max;
  double    peak_amp_x_row1;
  double    peak_amp_x_row2;
  double    peak_amp_x_col1;
  double    peak_amp_x_col2;
};

class ImgPeakFinderAB : public Module {
public:

    enum SELECTION_MODE{ SELECTION_OFF, SELECTION_ON, SELECTION_INV };
  
  // Default constructor
  ImgPeakFinderAB (const std::string& name) ;

  // Destructor
  virtual ~ImgPeakFinderAB () ;

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

    void init(Event& evt, Env& env);
    void instArrays();
    void procData(Event& evt);
    void collectStat(const double* data);
    void findPeaks();
    MedianResult evaluateSoNForPixel(unsigned r, unsigned c, const double* data);
    void iterateOverConnectedPixels(int r, int c, PeakWork& pw);
    bool peakSelector(PeakWork& pw);
    void printPeakWork(PeakWork& pw);
    void savePeakInVector(PeakWork& pw);
    void printVectorOfPeaks();
    void maskUpdateControl();
    void setSelectionMode();
    bool eventSelector();
    void printEventSelectionPars(Event& evt, bool isSelected);
    void evaluateVectorOfIndexesForMedian();
    void printMatrixOfIndexesForMedian();
    void printVectorOfIndexesForMedian();

    void printInputParameters();
    void printShapeParameters();
    void printEventRecord(Event& evt, std::string comment="");

    void resetForEventProcessing();
    void resetStatArrays();
    void resetSignalArrays();

    void procStatArrays();
    template <typename T>
      void saveImgArrayInFile(const std::string& fname, const T* arr); //, const unsigned& rows, const unsigned& cols);

    void getMaskFromFile();
    void printMaskStatistics();

    void doOperationsForSelectedEvent(Event& evt);
    void savePeaksInEvent(Event& evt);
    void savePeaksInFile(std::string& fname, std::vector<Peak> peaks);
    void printSelectionStatistics();
    void printSelectionStatisticsByCounter();
    void printJobSummary();

private:
  Pds::Src       m_src;              // source address of the data object
  Source         m_str_src;          // string with source name
  std::string    m_key;              // string with key name
  std::string    m_key_signal_out;   // string with key for signal cspad array (background subtracted by median algorithm) 
  std::string    m_key_peaks_out;    // string with key for found peaks in selected events
  std::string    m_maskFile_inp;     // [in]  file with mask 
  std::string    m_maskFile_out;     // [out] file with mask 
  std::string    m_fracFile_out;     // [out] file with pixel status info: fraction of noisy images (events)
  std::string    m_evtFile_out;      // [out] file name prefix for event array output
  float          m_rmin;             // radial parameter of the area for median algorithm   ~2-3
  float          m_dr;               // radial band width of the area for median algorithm  ~1-2
  float          m_SoNThr_noise;     // S/N threshold for noisy pixels  ~3
  float          m_SoNThr_signal;    // S/N threshold for signal pixels ~12
  float          m_frac_noisy_imgs;  // For hot-pixel mask definition

  unsigned       m_peak_npix_min;    // Peak selection parameters ~2-4
  unsigned       m_peak_npix_max;    //                           ~20-25
  double         m_peak_amp_tot_thr; // if >1 then ON, else OFF
  double         m_peak_SoN_thr;     // Peak S/N threshold

  unsigned       m_event_npeak_min;  // Minimum number of peaks in the event for selector ~10
  unsigned       m_event_npeak_max;  // Minimum number of peaks in the event for selector ~10000
  double         m_event_amp_tot_thr;// Amplitude threshold on total amplitude in all peaks. 0=off

  unsigned       m_nevents_mask_update;
  unsigned       m_nevents_mask_accum;

  std::string    m_sel_mode_str;
  SELECTION_MODE m_sel_mode;
  unsigned       m_out_file_bits;     // Yes/No bits to control output files
  unsigned       m_print_bits;   
  unsigned long  m_count;             // number of events from the beginning of job
  unsigned long  m_count_selected;    // number of selected events from the beginning of job
  unsigned long  m_count_mask_update; // number of events from the last mask update
  unsigned long  m_count_mask_accum;  // number of events from the beginning of the mask statistics accumulation

  unsigned       m_shape[2]; // image shape
  unsigned       m_rows;
  unsigned       m_cols;
  unsigned       m_size;
  int            m_rows1;
  int            m_cols1;

  unsigned*      m_stat           ; // pointer for 2-d array with m_size = m_rows * m_cols
  int16_t*       m_mask           ;
  float*         m_frac_noisy_evts;
  double*        m_signal         ;
  double*        m_bkgd           ;
  double*        m_noise          ;
  uint16_t*      m_proc_status    ;

  int16_t*       m_newdata;

  ImgAlgos::ImgParametersV1 *m_mask_initial;

  std::vector<TwoIndexes> v_indForMediane;

  std::vector<Peak> v_peaks;
  double    m_event_amp_tot;
  TimeInterval *m_time;
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGPEAKFINDERAB_H
