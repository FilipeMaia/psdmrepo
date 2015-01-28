#ifndef IMGALGOS_NDARRAVERAGE_H
#define IMGALGOS_NDARRAVERAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrAverage.
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
//#include "psddl_psana/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "ImgAlgos/GlobalMethods.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *
 * This module averages over events the per-pixel data of the ndarray<double,NDim>, where NDim=[1,5] 
 * and saves files for sum, averaged, rms, and mask if the file name(s) are specified. 
 * Input data can be specified by the source 
 * and key parameters. Averaging may have up to three stages, depending on configuration parameters:
 * 
 *     0-stage: the pixel intensities are averaged without any constrains for events from 0 
 *              to evts_stage1, the preliminary averaged and rms values are defined for each 
 *              pixel at the end of this stage.
 *     1-stage: starting from event evts_stage1 the pixel data are collected only for 
 *              abs(amplitude-average0) < gate_width1. At the end of this stage the 
 *              preliminary averaged and rms values are defined for each pixel.
 *     2-stage: starting from the event evts_stage1 + evts_stage2 the pixel data are 
 *              collected only for abs(amplitude-average1) < gate_width2. At the end 
 *              of this stage the preliminary averaged and rms values are defined for 
 *              each pixel and saved in the files specified by the avefile and rmsfile 
 *              parameters, respectively.
 *     This 3-stage averaging algorithm eliminates large statistical fluctuations 
 * in the pixel amplitude spectrum.
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

class NDArrAverage : public Module {
public:

  // Default constructor
  NDArrAverage (const std::string& name) ;

  // Destructor
  virtual ~NDArrAverage () ;

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
  void setFileMode();
  bool setCollectionMode(Event& evt);
  bool collectStat(Event& evt);
  void resetStatArrays();
  void procStatArrays();
  void saveArrInFile(std::string& fname, double* arr);
  void printEventRecord(Event& evt);
  void printSummaryForParser(Event& evt);
  void printStatBadPix();
  double evaluateThresholdOnRMS();

private:

  Pds::Src       m_src;             // source address of the data object
  Source         m_str_src;         // string with source name
  std::string    m_key;             // string with key name
  std::string    m_sumFile;
  std::string    m_aveFile;
  std::string    m_rmsFile;
  std::string    m_mskFile;
  std::string    m_hotFile;
  std::string    m_maxFile;
  std::string    m_file_type;       // File type "txt", "metatxt", or "bin"

  std::string    m_fname_ext;       // file name extension, for example for run 123: "-r0123.dat" 

  double         m_thr_rms;         // if rms > m_thr_rms - pixel is bad
  double         m_thr_min;         // if ave < m_thr_min - pixel is bad
  double         m_thr_max;         // if ave > m_thr_max - pixel is bad
  unsigned       m_print_bits;   
  unsigned long  m_count;           // number of found images
  unsigned long  m_count_ev;        // number of events from the beginning of job
  unsigned long  m_nev_stage1;
  unsigned long  m_nev_stage2;
  double         m_gate_width1;
  double         m_gate_width2;

  double         m_gate_width;

  /// Enumerated file type for "txt", "bin", etc.
  FILE_MODE      m_file_mode;

  bool           m_do_sum;
  bool           m_do_ave;
  bool           m_do_rms;
  bool           m_do_msk;
  bool           m_do_hot;
  bool           m_do_max;

  std::string    m_str_exp;
  std::string    m_str_run_num;
  std::string    m_str_source;

  NDArrPars*     m_ndarr_pars;
  unsigned       m_size;
  unsigned       m_nbadpix;

  unsigned*      m_stat;  // statistics per pixel
  double*        m_sum;   // sum per pixel
  double*        m_sum2;  // sum of squares per pixel
  double*        m_ave;   // average per pixel
  double*        m_rms;   // rms per pixel
  int*           m_msk;   // pixel mask per pixel; pixel is hot if rms > m_thr_rms, hot/cold = 0/1 
  int*           m_hot;   // hot-pixel mask per pixel (in style of Phil); pixel is hot if rms > m_thr_rms, hot/cold = 1/0 , 
  double*        m_max;   // maximal value over events per pixel

protected:

//-------------------

    template <typename T>
    void accumulateCorrelators(T* data)
    { 
      double amp(0);
      for (unsigned i=0; i<m_size; ++i) {

	amp = (double)data[i];
	if ( m_gate_width > 0 && std::abs(amp-m_ave[i]) > m_gate_width ) continue;

        m_stat[i] ++;
        m_sum [i] += amp;
        m_sum2[i] += amp*amp;
      }

      if (m_do_max) {
        for (unsigned i=0; i<m_size; ++i) {
	  amp = (double)data[i];
          if (amp > m_max[i]) m_max[i] = amp;
	}
      } // m_do_max
    }          

//-------------------

    template <typename T, unsigned NDim>
    bool collectStatForTypeNdim(Event& evt)
    { 
        shared_ptr< ndarray<T,NDim> > shp = evt.get(m_str_src, m_key, &m_src);
        if (shp.get()) { accumulateCorrelators<T>(shp->data()); return true; } 
        return false;
    }

//-------------------

    template <typename T>
    bool collectStatForType(Event& evt)
    { 
      unsigned ndim = m_ndarr_pars->ndim();

      // CONST
      if (ndim == 2 && collectStatForTypeNdim<const T,2>(evt)) return true;
      if (ndim == 3 && collectStatForTypeNdim<const T,3>(evt)) return true;
      if (ndim == 4 && collectStatForTypeNdim<const T,4>(evt)) return true;
      if (ndim == 5 && collectStatForTypeNdim<const T,5>(evt)) return true;
      if (ndim == 1 && collectStatForTypeNdim<const T,1>(evt)) return true;

      // NON-CONST
      if (ndim == 2 && collectStatForTypeNdim<T,2>(evt)) return true;
      if (ndim == 3 && collectStatForTypeNdim<T,3>(evt)) return true;
      if (ndim == 4 && collectStatForTypeNdim<T,4>(evt)) return true;
      if (ndim == 5 && collectStatForTypeNdim<T,5>(evt)) return true;
      if (ndim == 1 && collectStatForTypeNdim<T,1>(evt)) return true;

      return false;
    }

//-------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRAVERAGE_H
