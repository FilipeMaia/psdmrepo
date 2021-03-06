#ifndef IMGPIXSPECTRA_CAMERAPIXSPECTRA_H
#define IMGPIXSPECTRA_CAMERAPIXSPECTRA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraPixSpectra.
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
 *  @brief Creates the spectal array for all pixels in the Opal-1k, Princeton, etc camera-detectors.
 *
 *  CameraPixSpectra class is a psana module which creates and fills 
 *  the spectral array for all pixels in the Opal-1k, Princeton, etc array. 
 *  The spectral array has two dimensions, the total number of pixels and
 *  the number of amplitude bins requested in the list of configuration parameters.
 *
 *  An example of the configuration file (psana.cfg) for this module:
 *
 *    @code
 *    [psana]                                                                   
 *    files         = /reg/d/psdm/sxr/sxr16410/xtc/e75-r0081-s01-c00.xtc    
 *    modules       = ImgPixSpectra.CameraPixSpectra                        
 *                                                                          
 *    [ImgPixSpectra.CameraPixSpectra]                                      
 *    source        = SxrBeamline.0:Opal1000.1                              
 *    amin          =     0.                                                
 *    amax          =  1000.                                                
 *    nbins         =   100                                                 
 *    arr_fname     = camera-pix-spectra.txt            
 *    #events       =   500                                                  
 *    @endcode
 *
 *  The output file "camera-pix-spectra.txt" contains the spectral array 
 *  for CSPad pixels accumulated in job. 
 *  Axillary file with additional name extension ".sha" contains the shape parameters
 *  of the spectral array. The file(s) can be used for further analysis 
 *  or presentation, for example, using the python script:
 *
 *    @code
 *    ImgPixSpectra/data/PlotSpectralArrayFromFile.py camera-pix-spectra.txt
 *    @endcode
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id: CameraPixSpectra.h$
 *
 *  @author Mikhail S. Dubrovin
 */

class CameraPixSpectra : public Module {
public:

  // Default constructor
  CameraPixSpectra (const std::string& name) ;

  // Destructor
  virtual ~CameraPixSpectra () ;

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
  void arrayFill (const uint16_t* data);
  void arrayFill8(const uint8_t*  data);
  void saveArrayInFile();
  void saveShapeInFile();
  int  ampToIndex(double amp);
  
private:

  // Data members, this is for example purposes only

  //enum {m_npix_mini1 = 185 * 388 * 2};

  //enum {m_npix_mini1 = 1024}; // *1024};
  
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

  int           m_offset;
  int           m_width;
  int           m_height;
  int           m_numPixels;
  int           m_nbins1;
  double        m_factor;

  int*          m_arr;
};

} // namespace ImgPixSpectra

#endif // IMGPIXSPECTRA_CAMERAPIXSPECTRA_H
