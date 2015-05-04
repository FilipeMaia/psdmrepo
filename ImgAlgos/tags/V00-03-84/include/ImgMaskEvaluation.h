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

using namespace std;

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


protected:

//-------------------
/// Apply median algorithm for one pixel
      template <typename T>
      MedianResult
      evaluateSoNForPixelForType(unsigned ind1d, const T* data)
      {
        unsigned col = ind1d % m_cols;
        unsigned row = ind1d / m_cols;

        unsigned sum0 = 0;
        double   sum1 = 0;
        double   sum2 = 0;
        
        for( std::vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin(); 
                                                     ij != v_indForMediane.end(); ij++ ) { // v_indForMediane.begin();
          int ic = col + (ij->i);
          int ir = row + (ij->j);

          if(ic < 0)       continue;
          if(ic > m_cols1) continue;
          if(ir < 0)       continue;
          if(ir > m_rows1) continue;

          double amp = data[ir*m_cols + ic];
          sum0 ++;
          sum1 += amp;
          sum2 += amp*amp;
        }

        MedianResult res = {0,0,0,0};
        
        if ( sum0 > 0 ) {
          res.avg = sum1/sum0;                                // Averaged background level
          res.rms = std::sqrt( sum2/sum0 - res.avg*res.avg ); // RMS os the background around peak
          double dat = data[ind1d];
          double dif = dat - res.avg;
          res.sig = (dif>0) ? dif : 0;                        // Signal above the background
          res.SoN = (res.rms>0) ? res.sig/res.rms : 0;        // S/N ratio
        }

        return res;
      }

//-------------------

      template <typename T>
      bool collectStatForType(Event& evt) // const T* data, 
      { 
        shared_ptr< ndarray<const T,2> > img = evt.get(m_str_src, m_key, &m_src);
        if (img.get()) {

            const T* _data = img->data();
            double amp(0);

            if(m_do_mask_satu || m_do_frac_satu) 
              for (unsigned i=0; i<m_size; ++i) {
	        amp = _data[i];
	        if ( amp > m_thre_satu ) p_stat_satu[i] ++;
              }          

            if(m_do_mask_nois || m_do_frac_nois) 
              for (unsigned i=0; i<m_size; ++i) {
                if ( evaluateSoNForPixelForType<T>(i, _data).SoN > m_thre_nois ) p_stat_nois[i] ++;  
	      }

            return true;
        } 
        return false;
      }          

//-------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGMASKEVALUATION_H
