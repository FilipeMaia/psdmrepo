#ifndef IMGALGOS_IMGMASKEVALUATION_H
#define IMGALGOS_IMGMASKEVALUATION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgMaskEvaluation.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *
 * This module gets the image data array (ndarray<double,2>) and evaluates two masks:
 * 1) "saturated" mask for pixels, which had an intensity above the saturation-threshold at least once per run.
 * 2) "noise" mask for pixels, which cross the noise-threshold with friquency higher than specified.
 *
 * In the endJob this algorithm saves files (if the file name is specified) for:
 * 1) "saturated" mask,
 * 2) "noise" mask,
 * 3) "combined" mask. 
 * 4) fraction of noisy events,
 * 5) fraction of saturated events.
 *
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


class ImgMaskEvaluation : public Module {
public:

  // Default constructor
  ImgMaskEvaluation (const std::string& name) ;

  // Destructor
  virtual ~ImgMaskEvaluation () ;

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

    void printInputParameters();
    void initImgArrays(Event& evt);
    void resetStatArrays();
    void collectStat(Event& evt);
    void procStatArrays();
    void printEventRecord(Event& evt);
    void evaluateVectorOfIndexesForMedian();
    void printVectorOfIndexesForMedian();
    MedianResult evaluateSoNForPixel(unsigned ind1d, const double* data);

private:

  Pds::Src       m_src;            // source address of the data object
  Source         m_str_src;        // string with source name
  std::string    m_key;            // string with key name
  std::string    m_file_mask_satu; // string with file name
  std::string    m_file_mask_nois; // ...
  std::string    m_file_mask_comb; // ...
  std::string    m_file_frac_satu; // ...
  std::string    m_file_frac_nois; // ...
  double         m_thre_satu;      // treshold for pixel saturation
  double         m_thre_nois;      // treshold for pixel noise
  double         m_frac_satu;      // allowed fraction of saturated events
  double         m_frac_nois;      // allowed fraction of noisy events 
  unsigned       m_dr;             // radial size of the area for S/N evaluation
  unsigned       m_print_bits;     // control bits for printout
  unsigned long  m_count;          // number of events from the beginning of job

  bool           m_do_mask_satu;   // flag: true (if file available) = applay algorithm and save file
  bool           m_do_mask_nois;   // ...
  bool           m_do_mask_comb;   // ...
  bool           m_do_frac_satu;   // ...
  bool           m_do_frac_nois;   // ...
 
  unsigned       m_shape[2];       // image shape {rows, cols}
  unsigned       m_rows;
  unsigned       m_cols;
  int            m_rows1;
  int            m_cols1;
  unsigned       m_size;           // image size rows*cols 

  unsigned*      p_stat_satu;      // statistics per pixel for saturating events
  unsigned*      p_stat_nois;      // statistics per pixel for noisy events
  int16_t*       p_mask_satu;      // mask for saturated pixels 
  int16_t*       p_mask_nois;      // mask for noisy pixels  
  int16_t*       p_mask_comb;      // mask combined for saturated & noisy pixels  
  double*        p_frac_satu;      // fraction of saturated events
  double*        p_frac_nois;      // fraction of noisy events

  std::vector<TwoIndexes> v_indForMediane; // Vector of inexes for mediane algorithm

};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGMASKEVALUATION_H
