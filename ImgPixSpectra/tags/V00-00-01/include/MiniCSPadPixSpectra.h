#ifndef IMGPIXSPECTRA_MINICSPADPIXSPECTRA_H
#define IMGPIXSPECTRA_MINICSPADPIXSPECTRA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MiniCSPadPixSpectra.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgPixSpectra {

/// @addtogroup ImgPixSpectra

/**
 *  @ingroup ImgPixSpectra
 *
 *  @brief Creates the spectal array for all pixels in the Mini-CSPad detector.
 *
 *  MiniCSPadPixSpectra class is a psana module which creates and fills 
 *  the spectral array for all pixels in the Mini-CSPad array. The spectral
 *  array has two dimensions, the total number of pixels and
 *  the number of amplitude bins requested in the list of configuration parameters.
 *
 *  An example of the configuration file (psana.cfg) for this module:
 *
 *    @code
 *    [psana]
 *    files         = /reg/d/psdm/CXI/cxi49812/xtc/e157-r0203-s00-c00.xtc
 *    modules       = ImgPixSpectra.MiniCSPadPixSpectra
 *    
 *    [ImgPixSpectra.MiniCSPadPixSpectra]
 *    source        = CxiSc1.0:Cspad2x2.0
 *    amin          =   500.
 *    amax          =  1000.
 *    nbins         =   100
 *    arr_fname     = mini-cspad-pix-spectra.txt
 *    #events       =   500                                                  
 *    @endcode
 *
 *  The output file "mini-cspad-pix-spectra.txt" contains the spectral array 
 *  for CSPad pixels accumulated in job. 
 *  Axillary file with additional name extension ".sha" contains the shape parameters
 *  of the spectral array. The file(s) can be used for further analysis 
 *  or presentation, for example, using the python script:
 *
 *    @code
 *    ImgPixSpectra/data/PlotSpectralArrayFromFile.py mini-cspad-pix-spectra.txt
 *    @endcode
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id: MiniCSPadPixSpectra.h$
 *
 *  @author Mikhail S. Dubrovin
 */

class MiniCSPadPixSpectra : public Module {
public:

  // Default constructor
  MiniCSPadPixSpectra (const std::string& name) ;

  // Destructor
  virtual ~MiniCSPadPixSpectra () ;

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

  void printInputPars();
  void arrayInit();
  void arrayDelete();
  void arrayFill(const int16_t* data);
  void saveArrayInFile();
  void saveShapeInFile();
  int  ampToIndex(double amp);
  
private:

  // Data members, this is for example purposes only

  enum {m_npix_mini1 = 185 * 388 * 2};
  
  Source        m_src;         // Data source set from config file
  Pds::Src      m_actualSrc;
  std::string   m_key;
  double        m_amin;
  double        m_amax;
  int           m_nbins;
  std::string   m_arr_fname;
  std::string   m_arr_shape_fname;
  unsigned      m_maxEvents;
  bool          m_filter;
  long          m_count;

  int           m_nbins1;
  double        m_factor;

  int*          m_arr;
};

} // namespace ImgPixSpectra

#endif // IMGPIXSPECTRA_MINICSPADPIXSPECTRA_H
