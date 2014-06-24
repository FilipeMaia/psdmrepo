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
//#include <omp.h>

//----------------------
// Base Class Headers --
//----------------------
#include "ImgAlgos/CSPadBaseModule.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"
#include "ImgAlgos/CSPadMaskV1.h"
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
  int    quad;
  int    sect; 
  double col;
  double row; 
  double sigma_col;
  double sigma_row;
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
  double    peak_amp_x_col1;
  double    peak_amp_x_col2;
  double    peak_amp_x_row1;
  double    peak_amp_x_row2;
};

class CSPadArrPeakFinder : public CSPadBaseModule {
public:

    const static int SIZE_OF_ARRAY = MaxQuads * MaxSectors * SectorSize; 
    const static int NumColumns1   = NumColumns - 1;
    const static int NumRows1      = NumRows    - 1;

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
    void procData(Event& evt);
    void fillOutputArr(unsigned quad, int16_t* newdata);
    void makeVectorOfSectorAndIndexInArray(unsigned quad);
    void collectStatInQuad(unsigned quad, const int16_t* data);
    void collectStatInSect(unsigned quad, unsigned sect, const int16_t* sectData);
    void findPeaksInSect(unsigned quad, unsigned sect);
    MedianResult evaluateSoNForPixel(unsigned ic,unsigned ir,const int16_t* sectData);
    void iterateOverConnectedPixels(int quad, int sect, int ic, int ir, PeakWork& pw);
    bool peakSelector(PeakWork& pw);
    void printPeakWork(PeakWork& pw);
    void savePeakInVector(int quad, int sect, PeakWork& pw);
    void makeUnitedPeakVector();
    void printVectorOfPeaks();
    //void printPeak(int i, Peak* p);
    void maskUpdateControl();
    void setSelectionMode();
    bool eventSelector();
    void printEventSelectionPars(Event& evt, bool isSelected);
    void evaluateVectorOfIndexesForMedian();
    void printMatrixOfIndexesForMedian();
    void printVectorOfIndexesForMedian();

    void printInputParameters();
    void printEventId(Event& evt);
    void printTimeStamp(Event& evt, std::string comment="");

    void resetForEventProcessing();
    void resetStatArrays();
    void resetSignalArrays();

    void procStatArrays();
    template <typename T>
    void saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows]);

    void getMaskFromFile();
    void printInitialMaskStatistics();

    std::string strEventCounter();
    std::string strTimeStamp(Event& evt);
    std::string strRunNumber(Event& evt);
    void doOperationsForSelectedEvent(Event& evt);
    void savePeaksInEvent(Event& evt);
    void savePeaksInEventAsNDArr(Event& evt);
    void savePeaksInFile(std::string& fname, std::vector<Peak> peaks);
    void printSelectionStatistics();
    void printSelectionStatisticsByCounter();
    void printJobSummary();

private:
  std::string    m_key_signal_out;   // string with key for signal cspad array (background subtracted by median algorithm) 
  std::string    m_key_peaks_out;    // string with key for found peaks (vector<Peak>) in selected events
  std::string    m_key_peaks_nda;    // string with key for found peaks (ndarray<float,2>)in selected events
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

  unsigned       m_stat            [MaxQuads][MaxSectors][NumColumns][NumRows];
  int16_t        m_mask            [MaxQuads][MaxSectors][NumColumns][NumRows];
  float          m_frac_noisy_evts [MaxQuads][MaxSectors][NumColumns][NumRows];
  int16_t        m_signal          [MaxQuads][MaxSectors][NumColumns][NumRows];
  int16_t        m_bkgd            [MaxQuads][MaxSectors][NumColumns][NumRows];
  double         m_noise           [MaxQuads][MaxSectors][NumColumns][NumRows];
  uint16_t       m_proc_status     [MaxQuads][MaxSectors][NumColumns][NumRows];

  int16_t*       m_newdata;

  float          m_common_mode[MaxSectors];

  ImgAlgos::CSPadMaskV1 *m_mask_initial;

  std::vector<TwoIndexes> v_indForMediane;


  std::vector<TwoIndexes> v_sectAndIndexInArray[MaxQuads];
  std::vector<Peak> v_peaks;
  std::vector<Peak> v_peaks_in_sect[MaxQuads][MaxSectors];
  unsigned  m_quad;
  unsigned  m_sect;
  double    m_event_amp_tot;
  TimeInterval *m_time;

  //omp_lock_t *m_lock;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRPEAKFINDER_H
