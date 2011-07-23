//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPad...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/ImageCSPad.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
// to work with detector data include corresponding 
// header from psddl_psana package
// #include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/cspad.ddl.h"

#include "CSPadImage/Image2D.h"
#include "CSPadImage/ImageCSPad2x1.h"
#include "CSPadImage/QuadParameters.h"
#include "CSPadImage/ImageCSPadQuad.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <string.h>
//#include <memory>
//using namespace std;
//#include <stdio.h>
//#include <stdlib.h>

//#include <boost/lexical_cast.hpp>
//#include <sstream> // for int to string conversion using std::stringstream
//#include <iomanip> // for formatted conversion std::setw(3) , std::setfill

//#include <iostream>
//#include <string>
//using namespace std;

// This declares this class as psana module
using namespace CSPadImage;
PSANA_MODULE_FACTORY(ImageCSPad)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadImage {

//----------------
// Constructors --
//----------------
ImageCSPad::ImageCSPad (const std::string& name)
  : Module(name)
  , m_calibDir()
  , m_typeGroupName()
  , m_source()
  , m_src()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_calibDir      = configStr("calibDir",      "/reg/d/psdm/CXI/cxi35711/calib");
  m_typeGroupName = configStr("typeGroupName", "CsPad::CalibV1");
  m_source        = configStr("source",        "CxiDs1.0:Cspad.0");
  m_runNumber     = config   ("runNumber",     32U);
  m_maxEvents     = config   ("events",        32U);
  m_filter        = config   ("filter",        false);
  m_src           = m_source;

  this -> printInputPars();
}

//--------------
// Destructor --
//--------------
ImageCSPad::~ImageCSPad () {}

//----------------------------------------------------------

/// Method which is called once at the beginning of the job
//void ImageCSPad::beginJob(Event& evt, Env& env) {}

//----------------------------------------------------------
/// Method which is called at the beginning of the run
void ImageCSPad::beginRun(Event& evt, Env& env) 
{
  shared_ptr<Psana::CsPad::ConfigV3> config = env.configStore().get(m_src);

  cout << "ImageCSPad::beginRun " << endl;

  if (config.get()) {

    /*    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV3:";
      str << "\n  concentratorVersion = " << config->concentratorVersion();
      str << "\n  runDelay            = " << config->runDelay();
      str << "\n  eventCode           = " << config->eventCode();
      str << "\n  inactiveRunMode     = " << config->inactiveRunMode();
      str << "\n  activeRunMode       = " << config->activeRunMode();
      str << "\n  tdi                 = " << config->tdi();
      str << "\n  payloadSize         = " << config->payloadSize();
      str << "\n  badAsicMask0        = " << config->badAsicMask0();
      str << "\n  badAsicMask1        = " << config->badAsicMask1();
      str << "\n  asicMask            = " << config->asicMask();
      str << "\n  quadMask            = " << config->quadMask();
      str << "\n  numAsicsRead        = " << config->numAsicsRead();
      str << "\n  numSect             = " << config->numSect();
      str << "\n  numQuads            = " << config->numQuads();

      std::vector<int> v_quads_shape = config->quads_shape();
      str << "\n  v_quads_shape.size()= " << v_quads_shape.size();
      str << "\n  v_quads_shape[0]    = " << v_quads_shape[0];

      for (uint32_t q = 0; q < config->numQuads(); ++ q) {
        str << "\n  roiMask       (" << q << ") = " << config->roiMask(q);
        str << "\n  numAsicsStored(" << q << ") = " << config->numAsicsStored(q);
      }

      str << "\n  MaxQuadsPerSensor = " << Psana::CsPad::MaxQuadsPerSensor;
      str << "\n  ASICsPerQuad      = " << Psana::CsPad::ASICsPerQuad;
      str << "\n  RowsPerBank       = " << Psana::CsPad::RowsPerBank;
      str << "\n  FullBanksPerASIC  = " << Psana::CsPad::FullBanksPerASIC;
      str << "\n  BanksPerASIC      = " << Psana::CsPad::BanksPerASIC;
      str << "\n  ColumnsPerASIC    = " << Psana::CsPad::ColumnsPerASIC;
      str << "\n  MaxRowsPerASIC    = " << Psana::CsPad::MaxRowsPerASIC;
      str << "\n  PotsPerQuad       = " << Psana::CsPad::PotsPerQuad;
      str << "\n  TwoByTwosPerQuad  = " << Psana::CsPad::TwoByTwosPerQuad;
      str << "\n  SectorsPerQuad    = " << Psana::CsPad::SectorsPerQuad;
    }  // WithMsgLog(...)
    */

      for (uint32_t q = 0; q < config->numQuads(); ++ q) {
        m_roiMask[q]         = config->roiMask(q);
        m_numAsicsStored[q]  = config->numAsicsStored(q);
      }
  }  // if (config.get())

  m_cspad_calibpar = new CSPadCalibPars("calibname");

  this -> fillQuadXYmin();
}

//----------------------------------------------------------

/// Method which is called at the beginning of the calibration cycle
//void ImageCSPad::beginCalibCycle(Event& evt, Env& env) {}

//----------------------------------------------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void ImageCSPad::event(Event& evt, Env& env)
{
  // get event ID

  /*
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (not eventId.get()) {
    MsgLog(name(), info, "event ID not found");
  } else {
    MsgLog(name(), info, "event ID: " << *eventId);
  }
  */


  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {

    /*
    WithMsgLog(name(), info, str) {
      str << "CsPad::DataV2:";
      int nQuads = data2->quads_shape()[0];
        str << "\n    nQuads = " << nQuads ;
      for (int q = 0; q < nQuads; ++ q) {

        const Psana::CsPad::ElementV2& el = data2->quads(q);
        str << "\n  Element #" << q ;
        str << "\n    virtual_channel = " << el.virtual_channel() ;
        str << "\n    lane = "            << el.lane() ;
        str << "\n    tid = "             << el.tid() ;
        str << "\n    acq_count = "       << el.acq_count() ;
        str << "\n    op_code = "         << el.op_code() ;
        str << "\n    quad = "            << el.quad() ;
        str << "\n    seq_count = "       << el.seq_count() ;
        str << "\n    ticks = "           << el.ticks() ;
        str << "\n    fiducials = "       << el.fiducials() ;
        str << "\n    frame_type = "      << el.frame_type() ;
        str << "\n    sb_temp = [ ";

        const uint16_t* sb_temp = el.sb_temp();
        std::copy(sb_temp, sb_temp+10, // +Psana::CsPad::ElementV2::Nsbtemp, 
            std::ostream_iterator<uint16_t>(str, " "));
        str << "]";

        std::vector<int> v_image_shape = el.data_shape();
        str << "\n    v_image_shape = "   << v_image_shape[0]
                                  << "  " << v_image_shape[1]
                                  << "  " << v_image_shape[2];
      }
    } //WithMsgLog(
    */

    int nQuads = data2->quads_shape()[0];
    cout << "\n    nQuads = " << nQuads << endl;
    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);
        cout  << "\n  Quad / Element #" << q << endl;

	const uint16_t* data = el.data();
        int   quad           = el.quad() ;

        std::vector<int> v_image_shape = el.data_shape();

	QuadParameters *quadpars = new QuadParameters(quad, v_image_shape, 850, 850, m_numAsicsStored[q], m_roiMask[q]);
	//quadpars -> print();

        ImageCSPadQuad<uint16_t>* image_quad = new ImageCSPadQuad<uint16_t> (data, quadpars, m_cspad_calibpar);

	this -> addQuadToCSPadImage(image_quad, quadpars, m_cspad_calibpar);

    } //   for (int q ...
  } // if (data2.get())


  m_cspad_image_2d = new Image2D<float>(&m_cspad_image[0][0],NRowsDet,NColsDet); 

  this -> saveCSPadImageInFile();

  // this is how to skip event (all downstream modules will not be called)
  if (m_filter && m_count % 10 == 0) skip();
 

  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) stop();
  // increment event counter
  m_count++;
  cout << "m_count=" << m_count << endl;

}

//----------------------------------------------------------

void ImageCSPad::addQuadToCSPadImage(ImageCSPadQuad<uint16_t> *image_quad, QuadParameters *quadpars, CSPadCalibPars *cspad_calibpar)
{
   cout << "ImageCSPad::addQuadToCSPadImage()" << endl;

            Image2D<uint16_t> *quad_image2d = image_quad -> getQuadImage2D();

            uint32_t quad = quadpars -> getQuadNumber();
	    uint32_t rot_index = (int)(cspad_calibpar -> getQuadRotation(quad)/90);

            size_t ncols = quad_image2d -> getNCols(rot_index);
            size_t nrows = quad_image2d -> getNRows(rot_index);

	    uint32_t ixmin = (uint32_t)m_xmin_quad[quad];
	    uint32_t iymin = (uint32_t)m_ymin_quad[quad];

	    for (uint32_t row=0; row<nrows; row++) {
	    for (uint32_t col=0; col<ncols; col++) {

               m_cspad_image[ixmin+row][iymin+col] = quad_image2d -> rotN90(row, col, rot_index);

	    }
	    }
}

//----------------------------------------------------------
// Define m_xmin_quad [q], m_ymin_quad [q] from a bunch of calibration parameters 

void ImageCSPad::fillQuadXYmin()
{
   cout << "ImageCSPad::fillQuadXYmin()" << endl;

	    float margX  = m_cspad_calibpar -> getMargX  ();
	    float margY  = m_cspad_calibpar -> getMargY  ();
	    float gapX   = m_cspad_calibpar -> getGapX   ();
	    float gapY   = m_cspad_calibpar -> getGapY   ();
	    float shiftX = m_cspad_calibpar -> getShiftX ();
	    float shiftY = m_cspad_calibpar -> getShiftY ();

	    // self.quadXOffset = [ margX+0-gapX+shiftX,  margX+  0+0-gapX-shiftX,  margX+834-2+gapX-shiftX,  margX+834+0+gapX+shiftX]
	    // self.quadYOffset = [ margY+3-gapY-shiftY,  margY+834-1+gapY-shiftY,  margY+834-5+gapY+shiftY,  margY+  0+2-gapY+shiftY]

	    float dx[] = {margX-gapX+shiftX,  margX-gapX-shiftX,  margX+gapX-shiftX,  margX+gapX+shiftX};
	    float dy[] = {margY-gapY-shiftY,  margY+gapY-shiftY,  margY+gapY+shiftY,  margY-gapY+shiftY};

            for (int q=0; q<4; q++) {

                m_xmin_quad [q] = m_cspad_calibpar -> getOffsetX    (q) 
                                + m_cspad_calibpar -> getOffsetCorrX(q)  
                                + dx[q];

                m_ymin_quad [q] = m_cspad_calibpar -> getOffsetY    (q) 
                                + m_cspad_calibpar -> getOffsetCorrY(q) 
                                + dy[q];	    
	    }
}

//----------------------------------------------------------

/// Method which is called at the end of the calibration cycle
//void ImageCSPad::endCalibCycle(Event& evt, Env& env) {}

//----------------------------------------------------------

/// Method which is called at the end of the run
//void ImageCSPad::endRun(Event& evt, Env& env) {}

//----------------------------------------------------------

/// Method which is called once at the end of the job
void ImageCSPad::endJob(Event& evt, Env& env) {

  WithMsgLog(name(), info, str) { str << "\nImageCSPad::endJob\n"; }

  // testOfImageClasses();
}

//----------------------------------------------------------

void ImageCSPad::printInputPars()
{
  WithMsgLog(name(), info, str) { 
  str << "\nInput parameters:";
  str << "\ncalibDir      :" << m_calibDir;
  str << "\ntypeGroupName :" << m_typeGroupName;
  str << "\nsource        :" << m_source;
  str << "\nrunNumber     :" << m_runNumber;
  str << "\nmaxEvents     :" << m_maxEvents;
  str << "\nfilter        :" << m_filter;
  }
}

//----------------

void ImageCSPad::saveCSPadImageInFile()
{
    string fname = "image_cspad.txt";
    m_cspad_image_2d -> saveImageInFile(fname,0);
}

//----------------

//----------------------------------------------------------

/// Method which is called once at the end of the job
void ImageCSPad::testOfImageClasses() {

  uint16_t arr2d[3][4] = { { 0, 1, 2, 3 },
                           { 4, 5, 6, 7 },
                           { 8, 9, 10,11} 
                         };
  
		     int nrows=3;
		     int ncols=4;

  Image2D<uint16_t>* test_image = new Image2D<uint16_t>(&arr2d[0][0],nrows,ncols);  
                     test_image -> printEntireImage();
                     test_image -> printEntireImage(1);
                     test_image -> printEntireImage(2);
                     test_image -> printEntireImage(3);


		     int gap_ncols = 2;

  ImageCSPad2x1<uint16_t>* image_2x1 = new ImageCSPad2x1<uint16_t>(&arr2d[0][0], gap_ncols, nrows, ncols);  
	                   image_2x1 -> printEntireImage(0);
	                   image_2x1 -> printEntireImage(1);
	                   image_2x1 -> printEntireImage(2);
	                   image_2x1 -> printEntireImage(3);

	                   image_2x1 -> saveImageInFile("image.txt",1);
}


//----------------------------------------------------------

} // namespace CSPadImage
