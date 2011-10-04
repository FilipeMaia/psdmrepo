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
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "psddl_psana/acqiris.ddl.h"
#include "PSEvt/EventId.h"

#include "CSPadPixCoords/Image2D.h"

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

//--------------------

/// Method which is called once at the beginning of the job
void 
PixCoordsTest::beginJob(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the run
void 
PixCoordsTest::beginRun(Event& evt, Env& env)
{
  cout << "ImageCSPad::beginRun " << endl;

  //m_cspad_calibpar = new PSCalib::CSPadCalibPars(); // get default calib pars from my local directory.
  m_cspad_calibpar   = new PSCalib::CSPadCalibPars(m_calibDir, m_typeGroupName, m_source, m_runNumber);
  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar );
  m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar );

  m_cspad_calibpar -> printCalibPars();
  m_pix_coords_2x1 -> print_member_data();

  this -> getQuadConfigPars(env);
}

//--------------------

void 
PixCoordsTest::getQuadConfigPars(Env& env)
{
  shared_ptr<Psana::CsPad::ConfigV3> config = env.configStore().get(m_src);
  if (config.get()) {
      for (uint32_t q = 0; q < config->numQuads(); ++ q) {
        m_roiMask[q]         = config->roiMask(q);
        m_numAsicsStored[q]  = config->numAsicsStored(q);
      }
  }

  m_n2x1         = Psana::CsPad::SectorsPerQuad;     // v_image_shape[0]; // 8
  m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // v_image_shape[1]; // 185
  m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // v_image_shape[2]; // 388
  m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;                               // 185*388;

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
PixCoordsTest::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

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

        this -> test_cspad_init ();

    int nQuads = data2->quads_shape()[0];
    cout << "nQuads = " << nQuads << endl;
    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);

        const int16_t* data = el.data();
        int            quad = el.quad() ;

        std::vector<int> v_image_shape = el.data_shape();

        CSPadPixCoords::QuadParameters *quadpars = new CSPadPixCoords::QuadParameters(quad, v_image_shape, NX_QUAD, NY_QUAD, m_numAsicsStored[q], m_roiMask[q]);

	this -> test_2x1  (data, quadpars, m_cspad_calibpar);
	this -> test_quad (data, quadpars, m_cspad_calibpar);

             struct timespec start, stop;
             int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time
	this -> test_cspad (data, quadpars, m_cspad_calibpar);
                 status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
        cout << "Time to fill quad is " 
             << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
             << " sec" << endl;
    }

        this -> test_cspad_save ();

  } // if (data2.get())
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
PixCoordsTest::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
PixCoordsTest::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
PixCoordsTest::endJob(Event& evt, Env& env)
{
}

//--------------------

void
PixCoordsTest::test_2x1(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar)
{
        int              quad           = quadpars -> getQuadNumber();
        uint32_t         roiMask        = quadpars -> getRoiMask();
	std::vector<int> v_image_shape  = quadpars -> getImageShapeVector();

        //cout  << "PixCoordsTest::test_2x1:  quadNumber=" << quad << endl;

  // SELECT QUADRANT NUMBER FOR TEST HERE !!!
  if(quad != 2) return;
 
  // SELECT 2x1 ORIENTATION FOR TEST HERE !!!
  CSPadPixCoords::PixCoords2x1::ORIENTATION rot=CSPadPixCoords::PixCoords2x1::R000;

  // SELECT 2x1 SECTION NUMBER FOR TEST HERE !!!
  uint32_t sect=1;

        bool bitIsOn = roiMask & (1<<sect);
        if( !bitIsOn ) return;
 
        // Initialization
        for (unsigned ix=0; ix<NX_2x1; ix++){
        for (unsigned iy=0; iy<NY_2x1; iy++){
          m_arr_2x1_image[ix][iy] = 0;
        }
        }

        double  mrgx=20.1;
        double  mrgy=20.1;

        const int16_t *data2x1 = &data[sect * m_sizeOf2x1Img];
 
        for (uint32_t c=0; c<m_ncols2x1; c++) {
        for (uint32_t r=0; r<m_nrows2x1; r++) {

           double xcoor = m_pix_coords_2x1 -> getPixCoorRotN90_pix (rot, XCOOR, r, c);
           double ycoor = m_pix_coords_2x1 -> getPixCoorRotN90_pix (rot, YCOOR, r, c);

           int ix = (int) (mrgx + xcoor);
           int iy = (int) (mrgy + ycoor);

	   if(ix <  0)      continue;
	   if(iy <  0)      continue;
	   if(ix >= NX_2x1) continue;
	   if(iy >= NY_2x1) continue;

           m_arr_2x1_image[ix][iy] = (double)data2x1[c*m_nrows2x1+r];

        }
        }

    CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_2x1_image[0][0],NX_2x1,NY_2x1);
    img2d -> saveImageInFile("test_2x1.txt",0);
}

//--------------------

void
PixCoordsTest::test_quad(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar)
{
        int              quad           = quadpars -> getQuadNumber();
        uint32_t         roiMask        = quadpars -> getRoiMask();
	std::vector<int> v_image_shape  = quadpars -> getImageShapeVector();
 
        cout  << "PixCoordsTest::test_quad:  quadNumber=" << quad << endl;

        // Initialization
        for (unsigned ix=0; ix<NX_QUAD; ix++){
        for (unsigned iy=0; iy<NY_QUAD; iy++){
          m_arr_quad_image[ix][iy] = 0;
        }
        }

	for(uint32_t sect=0; sect < m_n2x1; sect++)
	{
             bool bitIsOn = roiMask & (1<<sect);
             if( !bitIsOn ) continue;
 
             const int16_t *data2x1 = &data[sect * m_sizeOf2x1Img];

             for (uint32_t c=0; c<m_ncols2x1; c++) {
             for (uint32_t r=0; r<m_nrows2x1; r++) {

               int ix = (int) m_pix_coords_quad -> getPixCoorRot000_pix (XCOOR, quad, sect, r, c);
               int iy = (int) m_pix_coords_quad -> getPixCoorRot000_pix (YCOOR, quad, sect, r, c);

	       if(ix <  0)       continue;
	       if(iy <  0)       continue;
	       if(ix >= NX_QUAD) continue;
	       if(iy >= NY_QUAD) continue;

               m_arr_quad_image[ix][iy] = (double)data2x1[c*m_nrows2x1+r];

             }
             }
	}

        string fname = "test_q";
               fname += boost::lexical_cast<string>( quad );
               fname += ".txt";
 
  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_quad_image[0][0],NX_QUAD,NY_QUAD);
  img2d -> saveImageInFile(fname,0);
}

//--------------------

void
PixCoordsTest::test_cspad_init()
{
  // Initialization
  for (unsigned ix=0; ix<NX_CSPAD; ix++){
  for (unsigned iy=0; iy<NY_CSPAD; iy++){
    m_arr_cspad_image[ix][iy] = 0;
  }
  }
  m_cspad_ind = 0;
  m_coor_x_pix = m_pix_coords_cspad -> getPixCoorArrX_pix();
  m_coor_y_pix = m_pix_coords_cspad -> getPixCoorArrY_pix();
  m_coor_x_int = m_pix_coords_cspad -> getPixCoorArrX_int();
  m_coor_y_int = m_pix_coords_cspad -> getPixCoorArrY_int();
}

//--------------------

void
PixCoordsTest::test_cspad(const int16_t* data, CSPadPixCoords::QuadParameters* quadpars, PSCalib::CSPadCalibPars *cspad_calibpar)
{
      //int              quad           = quadpars -> getQuadNumber();
        uint32_t         roiMask        = quadpars -> getRoiMask();
	std::vector<int> v_image_shape  = quadpars -> getImageShapeVector();

	for(uint32_t sect=0; sect < m_n2x1; sect++)
	{
	     bool bitIsOn = roiMask & (1<<sect);
	     if( !bitIsOn ) { m_cspad_ind += m_sizeOf2x1Img; continue; }
 
             const int16_t *data2x1 = &data[sect * m_sizeOf2x1Img];

             //cout  << "  add section " << sect << endl;	     
 
             for (uint32_t c=0; c<m_ncols2x1; c++) {
             for (uint32_t r=0; r<m_nrows2x1; r++) {

               // This access takes 14ms/quad
               //int ix = (int) m_pix_coords_cspad -> getPixCoor_pix (XCOOR, quad, sect, r, c);
               //int iy = (int) m_pix_coords_cspad -> getPixCoor_pix (YCOOR, quad, sect, r, c);

               // This access takes 8ms/quad
               //int ix = (int) m_coor_x_pix [m_cspad_ind];
               //int iy = (int) m_coor_y_pix [m_cspad_ind];

               // This access takes 6ms/quad
               int ix = m_coor_x_int [m_cspad_ind];
               int iy = m_coor_y_int [m_cspad_ind];
	       m_cspad_ind++;

	       if(ix <  0)        continue;
	       if(iy <  0)        continue;
	       if(ix >= NX_CSPAD) continue;
	       if(iy >= NY_CSPAD) continue;

               m_arr_cspad_image[ix][iy] = (double)data2x1[c*m_nrows2x1+r];
             }
             }
	}
}

//--------------------

void
PixCoordsTest::test_cspad_save()
{
  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_cspad_image[0][0],NX_CSPAD,NY_CSPAD);
  img2d -> saveImageInFile("test_cspad.txt",0);
}

//--------------------

} // namespace CSPadPixCoords
