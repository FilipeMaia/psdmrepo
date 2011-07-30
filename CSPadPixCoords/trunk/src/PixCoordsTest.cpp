//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsTest...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsTest.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "psddl_psana/acqiris.ddl.h"
#include "PSEvt/EventId.h"

#include "CSPadImage/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(PixCoordsTest)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------
PixCoordsTest::PixCoordsTest (const std::string& name)
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
}

//--------------
// Destructor --
//--------------
PixCoordsTest::~PixCoordsTest ()
{
}

/// Method which is called once at the beginning of the job
void 
PixCoordsTest::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
PixCoordsTest::beginRun(Event& evt, Env& env)
{
  cout << "ImageCSPad::beginRun " << endl;

  shared_ptr<Psana::CsPad::ConfigV3> config = env.configStore().get(m_src);

  if (config.get()) {
      for (uint32_t q = 0; q < config->numQuads(); ++ q) {
        m_roiMask[q]         = config->roiMask(q);
        m_numAsicsStored[q]  = config->numAsicsStored(q);
      }
  }

  m_cspad_calibpar   = new PSCalib::CSPadCalibPars();
  //m_cspad_calibpar = new PSCalib::CSPadCalibPars(m_calibDir, m_typeGroupName, m_source, m_runNumber);

  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar );
  m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar );

  m_pix_coords_2x1 -> print_member_data();

  //this -> fillQuadXYmin();
}

/// Method which is called at the beginning of the calibration cycle
void 
PixCoordsTest::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
PixCoordsTest::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  ++m_count;
  if (m_count >= m_maxEvents) stop();
  cout << "m_count=" << m_count << endl;


  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {



    int nQuads = data2->quads_shape()[0];
    cout << "nQuads = " << nQuads << endl;
    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);

        const uint16_t* data = el.data();
        int   quad           = el.quad() ;

        std::vector<int> v_image_shape = el.data_shape();

        CSPadImage::QuadParameters *quadpars = new CSPadImage::QuadParameters(quad, v_image_shape, 850, 850, m_numAsicsStored[q], m_roiMask[q]);
        //quadpars -> print();

        //ImageCSPadQuad<uint16_t>* image_quad = new ImageCSPadQuad<uint16_t> (data, quadpars, m_cspad_calibpar);
        //this -> addQuadToCSPadImage(image_quad, quadpars, m_cspad_calibpar);

        //cout << " quadNumber=" << quad << endl;


	this -> test_2x1  (data, quadpars, m_cspad_calibpar);
	this -> test_quad (data, quadpars, m_cspad_calibpar);
    }







  } // if (data2.get())


}
  
/// Method which is called at the end of the calibration cycle
void 
PixCoordsTest::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
PixCoordsTest::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
PixCoordsTest::endJob(Event& evt, Env& env)
{
}

//--------------------

void
PixCoordsTest::test_2x1(const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar)
{
  int quad = quadpars -> getQuadNumber();

  // SELECT QUADRANT NUMBER FOR TEST HERE !!!
  if(quad != 2) return;

        cout  << "PixCoordsTest::test_2x1"
	      << " quadNumber="       << quad
              << endl;

        uint32_t roiMask = quadpars -> getRoiMask();

	std::vector<int> v_image_shape  = quadpars -> getImageShapeVector();
 
      //uint32_t m_n2x1         = v_image_shape[0];                 // 8
        uint32_t m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // v_image_shape[1];        // 185
        uint32_t m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // v_image_shape[2];        // 388
        uint32_t m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;                                      // 185*388;

        unsigned mrgx=20;
        unsigned mrgy=20;

        CSPadPixCoords::PixCoords2x1::ORIENTATION rot=CSPadPixCoords::PixCoords2x1::R000;
        CSPadPixCoords::PixCoords2x1::COORDINATE  X  =CSPadPixCoords::PixCoords2x1::X;
        CSPadPixCoords::PixCoords2x1::COORDINATE  Y  =CSPadPixCoords::PixCoords2x1::Y;

        // Initialization
        enum{ NX=500, NY=500 };
        float arr_image[NY][NX];
        for (unsigned ix=0; ix<NX; ix++){
        for (unsigned iy=0; iy<NY; iy++){
          arr_image[ix][iy] = 0;
        }
        }


// SELECT 2x1 SECTION NUMBER FOR TEST HERE !!!
	uint32_t sect=1;

            bool bitIsOn = roiMask & (1<<sect);
            if( !bitIsOn ) return;
 
            const uint16_t *data2x1 = &data[sect * m_sizeOf2x1Img];
 
            for (uint32_t c=0; c<m_ncols2x1; c++) {
            for (uint32_t r=0; r<m_nrows2x1; r++) {

               int ix = mrgx + (int) m_pix_coords_2x1 -> getPixCoorRotN90_pix (rot, X, r, c);
               int iy = mrgy + (int) m_pix_coords_2x1 -> getPixCoorRotN90_pix (rot, Y, r, c);

               arr_image[ix][iy] = (float)data2x1[c*m_nrows2x1+r];

            }
            }

  CSPadImage::Image2D<float> *img2d = new CSPadImage::Image2D<float>(&arr_image[0][0],NY,NX);
  img2d -> saveImageInFile("test_2x1.txt",0);

}


//--------------------

void
PixCoordsTest::test_quad(const uint16_t* data, CSPadImage::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar)
{
  int quad = quadpars -> getQuadNumber();

        cout  << "PixCoordsTest::test_quad"
	      << " quadNumber="       << quad
              << endl;

        uint32_t roiMask = quadpars -> getRoiMask();

	std::vector<int> v_image_shape  = quadpars -> getImageShapeVector();
 
        uint32_t m_n2x1         = v_image_shape[0];                 // 8
        uint32_t m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // v_image_shape[1];        // 185
        uint32_t m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // v_image_shape[2];        // 388
        uint32_t m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;                                      // 185*388;

        unsigned mrgx=20;
        unsigned mrgy=20;

        //CSPadPixCoords::PixCoords2x1::ORIENTATION rot=CSPadPixCoords::PixCoords2x1::R180;
        CSPadPixCoords::PixCoords2x1::COORDINATE X = CSPadPixCoords::PixCoords2x1::X;
        CSPadPixCoords::PixCoords2x1::COORDINATE Y = CSPadPixCoords::PixCoords2x1::Y;

        // Initialization
        enum{ NX=900, NY=900 };
        float arr_image[NY][NX];
        for (unsigned ix=0; ix<NX; ix++){
        for (unsigned iy=0; iy<NY; iy++){
          arr_image[ix][iy] = 0;
        }
        }

	for(uint32_t sect=0; sect < m_n2x1; sect++)
	{
             bool bitIsOn = roiMask & (1<<sect);
             if( !bitIsOn ) return;
 
             const uint16_t *data2x1 = &data[sect * m_sizeOf2x1Img];

             cout  << "  add section " << sect << endl;	     
 
             for (uint32_t c=0; c<m_ncols2x1; c++) {
             for (uint32_t r=0; r<m_nrows2x1; r++) {

               int ix = mrgx + (int) m_pix_coords_quad -> getPixCoorRot000_pix (X, quad, sect, r, c);
               int iy = mrgy + (int) m_pix_coords_quad -> getPixCoorRot000_pix (Y, quad, sect, r, c);

               arr_image[ix][iy] = (float)data2x1[c*m_nrows2x1+r];

             }
             }
	}

        string fname = "rest_q";
               fname += boost::lexical_cast<string>( quad );
               fname += ".txt";
 
  CSPadImage::Image2D<float> *img2d = new CSPadImage::Image2D<float>(&arr_image[0][0],NY,NX);
  img2d -> saveImageInFile(fname,0);

}

//--------------------

} // namespace CSPadPixCoords
