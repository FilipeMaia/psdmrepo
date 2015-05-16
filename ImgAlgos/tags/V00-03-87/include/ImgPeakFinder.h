#ifndef IMGALGOS_IMGPEAKFINDER_H
#define IMGALGOS_IMGPEAKFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgPeakFinder.
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
#include "CSPadPixCoords/Image2D.h"
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

struct Peak{
   double x;
   double y; 
   double ampmax;
   double amptot;
   unsigned npix;
};
   // ? double s1;
   // ? double s2; 
   // ? double tilt_angle;  

class ImgPeakFinder : public Module {
public:

  // Default constructor
  ImgPeakFinder (const std::string& name) ;

  // Destructor
  virtual ~ImgPeakFinder () ;

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

  void   printInputParameters();
  void   setWindowRange();
  void   printWindowRange();
  void   initImage();
  void   smearImage();
  double smearPixAmp(size_t& r0, size_t& c0);
  double weight(int& dr, int& dc);
  void   evaluateWeights();
  void   printWeights();
  void   saveImageInFile0(Event& evt);
  void   saveImageInFile1(Event& evt);
  void   saveImageInFile2(Event& evt);
  string getCommonFileName(Event& evt);
  bool   getAndProcImage(Event& evt);
  bool   procImage(Event& evt);
  void   findPeaks(Event& evt);
  void   checkIfPixIsPeak(size_t& r0, size_t& c0);
  void   printPeakInfo(size_t& row, size_t& col, double& amp, double& amp_tot, unsigned& npix );
  void   printPeakInfo(Peak& p);
  void   savePeakInfo(size_t& row, size_t& col, double& amp, double& amp_tot, unsigned& npix );
  void   savePeaksInEvent(Event& evt);
  void   savePeaksInEventAsNDArr(Event& evt);
  void   savePeaksInFile(Event& evt);


  enum{ MAX_IMG_SIZE=2000*2000 };
  enum{ MARGIN=10, MARGIN1=11 };

  Pds::Src    m_src;
  Source      m_str_src;
  std::string m_key;
  std::string m_key_peaks_vec;
  std::string m_key_peaks_nda;
  double   m_thr_low;
  double   m_thr_high;
  double   m_sigma;    // smearing sigma in pixel size
  int      m_nsm;      // number of pixels for smearing [i0-m_nsm, i0+m_nsm]
  int      m_npeak;    // number of pixels for peak finding in [i0-m_npeak, i0+m_npeak]
  float    m_xmin;
  float    m_xmax;
  float    m_ymin;
  float    m_ymax;
  bool     m_finderIsOn;
  long     m_event;
  unsigned m_print_bits;
  long     m_count;
  long     m_selected;

  size_t   m_nrows;
  size_t   m_ncols;

  size_t   m_rowmin; 
  size_t   m_rowmax;
  size_t   m_colmin;
  size_t   m_colmax;

  ndarray<const double,2> *m_ndarr;
  CSPadPixCoords::Image2D<double> *m_img2d;
  CSPadPixCoords::Image2D<double> *m_work2d;

  double m_weights[MARGIN][MARGIN];
  double m_data_arr[MAX_IMG_SIZE];
  double m_work_arr[MAX_IMG_SIZE];
  TimeInterval *m_time;
  std::vector<Peak> v_peaks;
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGPEAKFINDER_H
