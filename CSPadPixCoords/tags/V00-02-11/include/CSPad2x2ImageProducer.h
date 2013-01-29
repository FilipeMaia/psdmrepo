#ifndef CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H
#define CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2ImageProducer.
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
//#include "PSCalib/CSPadCalibPars.h"
#include "CSPadPixCoords/PixCoords2x1.h"
#include "CSPadPixCoords/PixCoordsCSPad2x2.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "PSEvt/Source.h"


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief CSPad2x2ImageProducer produces the CSPad2x2 image for each event and add it to the event in psana framework.
 *
 *  CSPad2x2ImageProducer works in psana framework. It does a few operation as follows:
 *  1) get the pixel coordinates from PixCoords2x1 and PixCoordsCSPad2x2 classes,
 *  2) get data from the event,
 *  3) produce the Image2D object with CSPad image for each event,
 *  4) add the Image2D object in the event for further modules.
 *
 *  The CSPad2x2 image array currently is shaped as [400][400] pixels.
 *
 *  This class should not be used directly in the code of users modules. 
 *  Instead, it should be added as a module in the psana.cfg file with appropriate parameters.
 *  Then, the produced Image2D object can be extracted from event and used in other modules.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PixCoords2x1, PixCoordsQuad, PixCoordsCSPad, CSPadImageGetTest
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CSPad2x2ImageProducer : public Module {
public:

  // Default constructor
  CSPad2x2ImageProducer (const std::string& name) ;

  // Destructor
  virtual ~CSPad2x2ImageProducer () ;

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
  int  getRunNumber(Event& evt);
  void getConfigPars(Env& env);
  void cspad_image_init();
  void processEvent(Event& evt, Env& env);

  //void cspad_image_fill(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar);
  void cspad_image_fill(const ndarray<int16_t,3>& data);
  void cspad_image_save_in_file(const std::string &filename = "cspad_image.txt");
  void cspad_image_add_in_event(Event& evt);
  void printTimeStamp(Event& evt);

private:

  // Data members, this is for example purposes only

  //std::string m_calibDir;       // i.e. ./calib
  //std::string m_typeGroupName;  // i.e. CsPad::CalibV1
  //Source      m_src;         // Data source set from config file
  Source      m_str_src;        // i.e. CxiDs1.0:Cspad.0
  Pds::Src    m_actualSrc;
  std::string m_inkey; 
  std::string m_outimgkey;   // i.e. "CSPad:Image"
  bool        m_tiltIsApplied;
  unsigned    m_print_bits;
  long        m_count;

  uint32_t m_roiMask;
  uint32_t m_numAsicsStored;

  uint32_t m_n2x1;         // 2
  uint32_t m_ncols2x1;     // 185
  uint32_t m_nrows2x1;     // 388
  uint32_t m_sizeOf2x1Img; // 185*388;

  PSCalib::CSPadCalibPars        *m_cspad_calibpar;
  CSPadPixCoords::PixCoords2x1   *m_pix_coords_2x1;
  //CSPadPixCoords::PixCoordsQuad  *m_pix_coords_quad;
  //CSPadPixCoords::PixCoordsCSPad *m_pix_coords_cspad;
  CSPadPixCoords::PixCoordsCSPad2x2 *m_pix_coords_cspad2x2;

  CSPadPixCoords::PixCoords2x1::COORDINATE XCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE YCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE ZCOOR;
	
  uint32_t   m_cspad_ind;
  double    *m_coor_x_pix;
  double    *m_coor_y_pix;
  uint32_t  *m_coor_x_int;
  uint32_t  *m_coor_y_int;

  float      m_common_mode[2];

  enum{ NX_CSPAD2X2=400, 
        NY_CSPAD2X2=400 };
  double m_arr_cspad2x2_image[NX_CSPAD2X2][NY_CSPAD2X2];
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_CSPAD2X2IMAGEPRODUCER_H
