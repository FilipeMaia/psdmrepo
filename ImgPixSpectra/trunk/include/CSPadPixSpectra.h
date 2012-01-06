#ifndef IMGPIXSPECTRA_CSPADPIXSPECTRA_H
#define IMGPIXSPECTRA_CSPADPIXSPECTRA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadPixSpectra.
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
#include "psddl_psana/cspad.ddl.h"
//#include "ndarray/ndarray.h"

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
 *  @brief Creates the spectal array for all pixels in the CSPad detector.
 *
 *  CSPadPixSpectra class is a psana module which creates and fills 
 *  the spectral array for all pixels in the CSPad array. The spectral
 *  array has two dimensions, the total number of CSPad pixels and
 *  the number of amplitude bins requested in the list of configuration parameters.
 *
 *  An example of the configuration file (psana.cfg) for this module:
 *
 *    @code
 *    [psana]
 *    files         = /reg/d/psdm/CXI/cxi35711/xtc/e86-r0009-s00-c00.xtc
 *    modules       = ImgPixSpectra.CSPadPixSpectra
 *
 *    [ImgPixSpectra.CSPadPixSpectra]
 *    source        = CxiDs1.0:Cspad.0
 *    amin          =    10.
 *    amax          =  2010.
 *    nbins         =   100
 *    arr_fname     = cspad_spectral_array_cfg.txt
 *    @endcode
 *
 *  The output file "cspad_spectral_array_cfg.txt" contains the spectral array 
 *  for CSPad pixels accumulated in job. This file can be used for further analysis 
 *  or presentation, for example, using the python script:
 *
 *    @code
 *    ./Plot2DArrayFromFile.py cspad_spectral_array_cfg.txt
 *    @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version \$Id: CSPadPixSpectra.h$
 *
 *  @author Mikhail S. Dubrovin
 */

  typedef Psana::CsPad::DataV2    CSPadDataType;
  typedef Psana::CsPad::ElementV2 CSPadElementType;
  typedef Psana::CsPad::ConfigV3  CSPadConfigType;

class CSPadPixSpectra : public Module {
public:

  // Default constructor
  CSPadPixSpectra (const std::string& name) ;

  // Destructor
  virtual ~CSPadPixSpectra () ;

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

  void getQuadConfigPars(Env& env);
  void printQuadConfigPars();
  void printInputPars();
  void arrayInit();
  void arrayDelete();
  void loopOverQuads(shared_ptr<CSPadDataType> data);
  void arrayFill(int quad, const int16_t* data, uint32_t roiMask);
  void saveArrayInFile();
  int  ampToIndex(double amp);

private:

  // Data members, this is for example purposes only
  
  Source   m_src;         // Data source set from config file
  Pds::Src m_actualSrc;
  unsigned m_maxEvents;
  double   m_amin;
  double   m_amax;
  int      m_nbins;
  string   m_arr_fname;
  bool     m_filter;
  long     m_count;
  
  int      m_nbins1;
  double   m_factor;

  uint32_t m_roiMask        [4];
  uint32_t m_numAsicsStored [4];

  uint32_t m_nquads;         // 4
  uint32_t m_n2x1;           // 8
  uint32_t m_ncols2x1;       // 185
  uint32_t m_nrows2x1;       // 388
  uint32_t m_sizeOf2x1Arr;   // 185*388;
  uint32_t m_sizeOfQuadArr;  // 185*388*8;
  uint32_t m_sizeOfCSPadArr; // 185*388*8*4;
  uint32_t m_pixel_ind;

  //std::vector<int> m_image_shape;

  int*           m_arr;
  //ndarray<int,2> m_arr2d;
};

} // namespace ImgPixSpectra

#endif // IMGPIXSPECTRA_CSPADPIXSPECTRA_H
