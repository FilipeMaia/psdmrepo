#ifndef CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H
#define CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageProducer.
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

#include "PSEvt/Source.h"


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

class CSPadImageProducer : public Module {
public:

  // Default constructor
  CSPadImageProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPadImageProducer () ;

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

  void cspad_image_init();
  void cspad_image_fill (const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  void cspad_image_save_in_file(const std::string &filename = "cspad_image.txt");
  void cspad_image_add_in_event(Event& evt, const std::string &keyname = "CSPad:Image");

protected:

private:

  // Data members, this is for example purposes only

  std::string m_calibDir;       // i.e. /reg/d/psdm/CXI/cxi35711/calib
  std::string m_typeGroupName;  // i.e. CsPad::CalibV1
  std::string m_source;         // i.e. CxiDs1.0:Cspad.0
   
  Source   m_src;         // Data source set from config file
  unsigned m_runNumber;
  unsigned m_maxEvents;
  bool     m_filter;
  long     m_count;

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
	
  uint32_t  m_cspad_ind;
  float    *m_coor_x_pix;
  float    *m_coor_y_pix;
  uint32_t *m_coor_x_int;
  uint32_t *m_coor_y_int;

  enum{ NX_QUAD=850, 
        NY_QUAD=850 };

  enum{ NX_CSPAD=1750, 
        NY_CSPAD=1750 };
  float m_arr_cspad_image[NX_CSPAD][NY_CSPAD];
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPADIMAGEPRODUCER_H
