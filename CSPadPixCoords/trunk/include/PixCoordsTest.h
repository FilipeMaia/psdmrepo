#ifndef CSPADPIXCOORDS_PIXCOORDSTEST_H
#define CSPADPIXCOORDS_PIXCOORDSTEST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsTest.
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
#include "PSCalib/CSPadCalibPars.h"

#include "CSPadImage/QuadParameters.h"
#include "CSPadPixCoords/PixCoords2x1.h"
#include "CSPadPixCoords/PixCoordsQuad.h"
#include "CSPadPixCoords/PixCoordsCSPad.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class PixCoordsTest : public Module {
public:

  // Default constructor
  PixCoordsTest (const std::string& name) ;

  // Destructor
  virtual ~PixCoordsTest () ;

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

  void getQuadConfigPars(Env& env);

  void test_2x1   (const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  void test_quad  (const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  void test_cspad (const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  void test_cspad_init();
  void test_cspad_save();


protected:

private:

  // Data members, this is for example purposes only

  std::string m_calibDir;       // i.e. /reg/d/psdm/CXI/cxi35711/calib
  std::string m_typeGroupName;  // i.e. CsPad::CalibV1
  std::string m_source;         // i.e. CxiDs1.0:Cspad.0
   
  Source m_src;         // Data source set from config file
  unsigned m_runNumber;
  unsigned m_maxEvents;
  bool m_filter;
  long m_count;

  uint32_t m_roiMask        [4];
  uint32_t m_numAsicsStored [4];

  uint32_t m_n2x1;         // 8
  uint32_t m_ncols2x1;     // 185
  uint32_t m_nrows2x1;     // 388
  uint32_t m_sizeOf2x1Img; // 185*388;

  PSCalib::CSPadCalibPars        *m_cspad_calibpar;

  CSPadPixCoords::PixCoords2x1   *m_pix_coords_2x1;
  CSPadPixCoords::PixCoordsQuad  *m_pix_coords_quad;
  CSPadPixCoords::PixCoordsCSPad *m_pix_coords_cspad;

  CSPadPixCoords::PixCoords2x1::COORDINATE XCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE YCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE ZCOOR;

  enum{ NX_2x1=500, 
        NY_2x1=500 };
  float m_arr_2x1_image[NX_2x1][NY_2x1];

  enum{ NX_QUAD=850, 
        NY_QUAD=850 };
  float m_arr_quad_image[NX_QUAD][NY_QUAD];

  enum{ NX_CSPAD=1750, 
        NY_CSPAD=1750 };
  float m_arr_cspad_image[NX_CSPAD][NY_CSPAD];

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSTEST_H
