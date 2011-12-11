//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadInterpolImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPadInterpolImageProducer.h"

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
PSANA_MODULE_FACTORY(CSPadInterpolImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPadInterpolImageProducer::CSPadInterpolImageProducer (const std::string& name)
  : Module(name)
  , m_typeGroupName()
  , m_source()
  , m_src()
  , m_maxEvents()
  , m_filter()
  , m_tiltIsApplied()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_typeGroupName = configStr("typeGroupName", "CsPad::CalibV1");
  m_source        = configStr("source",        "CxiDs1.0:Cspad.0");
  m_maxEvents     = config   ("events",        32U);
  m_filter        = config   ("filter",        false);
  m_tiltIsApplied = config   ("tiltIsApplied", true);
  m_src           = m_source;
}


//--------------
// Destructor --
//--------------

CSPadInterpolImageProducer::~CSPadInterpolImageProducer ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPadInterpolImageProducer::beginJob(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the run
void 
CSPadInterpolImageProducer::beginRun(Event& evt, Env& env)
{
  cout << "ImageCSPad::beginRun " << endl;

  // get run number
  shared_ptr<EventId> eventId = evt.get();
  int run = 0;
  if (eventId.get()) {
    run = eventId->run();
  } else {
    MsgLog(name(), warning, "Cannot determine run number, will use 0.");
  }

  //m_cspad_calibpar = new PSCalib::CSPadCalibPars(); // get default calib pars from my local directory
                                                      // ~dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det/
  m_cspad_calibpar   = new PSCalib::CSPadCalibPars(env.calibDir(), m_typeGroupName, m_source, run);
  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar, m_tiltIsApplied );
  m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar, m_tiltIsApplied );

  m_cspad_calibpar  -> printCalibPars();
  //m_pix_coords_2x1  -> print_member_data();
  //m_pix_coords_quad -> print_member_data(); 

  this -> getConfigPars(env);

  this -> fill_address_table_1();
  this -> fill_address_and_weights_of_4_neighbors();
  cout << "CSPadInterpolImageProducer::beginRun: Initialization is done!!!\n";
}

//--------------------

void 
CSPadInterpolImageProducer::getConfigPars(Env& env)
{
  shared_ptr<Psana::CsPad::ConfigV3> config = env.configStore().get(m_src);
  if (config.get()) {
      for (uint32_t q = 0; q < config->numQuads(); ++ q) {
        m_roiMask[q]         = config->roiMask(q);
        m_numAsicsStored[q]  = config->numAsicsStored(q);
      }
  }

  m_nquads       = 4;
  m_n2x1         = Psana::CsPad::SectorsPerQuad;     // v_image_shape[0]; // 8
  m_ncols2x1     = Psana::CsPad::ColumnsPerASIC;     // v_image_shape[1]; // 185
  m_nrows2x1     = Psana::CsPad::MaxRowsPerASIC * 2; // v_image_shape[2]; // 388
  m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1;                               // 185*388;

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  m_addr_empty = (ArrAddr){-1,-1,-1,-1};
  cout << "TEST: m_addr_empty = " << m_addr_empty << endl;

}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
CSPadInterpolImageProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadInterpolImageProducer::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  ++m_count;
  if (m_count >= m_maxEvents) stop();

    //-------------------- Time
    struct timespec start, stop;
    int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time
    //-------------------- Time

  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src, "", &m_actualSrc); // get m_actualSrc here

  if (data2.get()) {

    bool  quadIsAvailable[] = {false, false, false, false};
    ndarray<int16_t, 3> data[4];
    QuadParameters *quadpars[4];
   
    int nQuads = data2->quads_shape()[0];

    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);
        int  quad  = el.quad();

        data[quad] = el.data();
        quadpars[quad] = new QuadParameters(quad, NX_QUAD, NY_QUAD, m_numAsicsStored[q], m_roiMask[q]);
        quadIsAvailable[quad] = true;

	//cout << " q = "                     << q 
	//     << " quad = "                  << quad 
	//     << " quadIsAvailable[quad] = " << quadIsAvailable[quad]
        //     << endl;
    }

    this -> cspad_image_init ();
    this -> cspad_image_interpolated_fill (data, quadpars, quadIsAvailable);
    this -> cspad_image_add_in_event(evt,"CSPad:Image");

    //-------------------- Time
    status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
    cout << "  Event: " << m_count 
         << "  Time to fill cspad is " 
         << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
         << " sec" << endl;
    //-------------------- Time

  } // if (data2.get())
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
CSPadInterpolImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
CSPadInterpolImageProducer::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
CSPadInterpolImageProducer::endJob(Event& evt, Env& env)
{
}

//--------------------
/**
  * Operator << is useful for printing of the struct ArrAddr objects.
  */ 
std::ostream&
operator<< (std::ostream& s, const ArrAddr& a)
{
    s << "ArrAddr {"
      << " quad=" << a.quad                    
      << " sect=" << a.sect 
      << " row="  << a.row
      << " col="  << a.col 
      << " }\n" ;
    return s ;
}

//--------------------
/**
  * Comparison of the two struct ArrAddr objects.
  */ 
bool areEqual( const ArrAddr& a1, const ArrAddr& a2 ) 
{
  if (    a1.quad == a2.quad 
       && a1.sect == a2.sect 
       && a1.row  == a2.row 
       && a1.col  == a2.col
      ) return true;
  else  return false;
}

//--------------------
/**
  * Initialization of the table of addresses between the CSPad and 2x1 coordinates.
  */ 
void
CSPadInterpolImageProducer::init_address_table_1()
{
    // Initialization

    for (unsigned ix=0; ix<NX_CSPAD; ix++){
    for (unsigned iy=0; iy<NY_CSPAD; iy++){

        m_address_table_1[ix][iy] = m_addr_empty;
    }
    }
    // cout << "TEST: m_address_table_1[100][100] = " << m_address_table_1[100][100] << endl;
}

//--------------------
/**
  * Fills the table of addresses between the CSPad and 2x1 coordinates.
  */ 
void
CSPadInterpolImageProducer::fill_address_table_1()
{
    this -> init_address_table_1();

    for (uint32_t q=0; q < m_nquads;   q++) { 
    for (uint32_t s=0; s < m_n2x1;     s++) {
    for (uint32_t c=0; c < m_ncols2x1; c++) {
    for (uint32_t r=0; r < m_nrows2x1; r++) {

        double x = m_pix_coords_cspad -> getPixCoor_pix (XCOOR, q, s, r, c);
        double y = m_pix_coords_cspad -> getPixCoor_pix (YCOOR, q, s, r, c);
        int ix = (int)x + 1;
        int iy = (int)y + 1;

	//cout << "x:ix, y:iy =" << ix << " : " << x << "      " << iy << " : " << y << endl;

        if(ix <  0)        continue;
        if(iy <  0)        continue;
        if(ix >= NX_CSPAD) continue;
        if(iy >= NY_CSPAD) continue;

        m_address_table_1[ix][iy] = (ArrAddr){q, s, r, c};

    } // rows 
    } // cols
    } // sect
    } // quad

    cout << "TEST: m_address_table_1[100][100] = " << m_address_table_1[100][100] << endl;
}

//--------------------
/**
  * Initialization of the tables of addresses and weights
  * of the 4 neighbors from 2x1 matrix surrounding each element of the
  * uniform CSPad matrix.
  */ 
void
CSPadInterpolImageProducer::init_address_and_weights_of_4_neighbors()
{
    for (unsigned ix=0; ix<NX_CSPAD; ix++){
    for (unsigned iy=0; iy<NY_CSPAD; iy++){
    for (unsigned ip=0; ip<4;        ip++){

	m_address[ix][iy][ip] = m_addr_empty;
	m_weight [ix][iy][ip] = 0;

    } // ip
    } // iy
    } // ix
}

//--------------------
/**
  * Filling of the tables of addresses and weights
  * of the 4 neighbors from 2x1 matrix surrounding each element of the
  * uniform CSPad matrix.
  */ 
void
CSPadInterpolImageProducer::fill_address_and_weights_of_4_neighbors()
{
    this -> init_address_and_weights_of_4_neighbors();

    for (unsigned ix=0; ix<NX_CSPAD; ix++){
    for (unsigned iy=0; iy<NY_CSPAD; iy++){

      this -> get_address_of_4_neighbors(ix,iy);
    }
    }

    for (unsigned ix=0; ix<NX_CSPAD; ix++){
    for (unsigned iy=0; iy<NY_CSPAD; iy++){

      this -> get_weight_of_4_neighbors(ix,iy);
    }
    }

    cout << "TEST: m_address[100][100][...] : \n" 
         << "      [0]:" << m_address[100][100][0] 
         << "      [1]:" << m_address[100][100][1] 
         << "      [2]:" << m_address[100][100][2] 
         << "      [3]:" << m_address[100][100][3] 
         << endl;

    cout << "TEST: m_weight[100][100][...] : " 
         << " [0]:" << m_weight[100][100][0] 
         << " [1]:" << m_weight[100][100][1] 
         << " [2]:" << m_weight[100][100][2] 
         << " [3]:" << m_weight[100][100][3] 
         << endl;
}

//--------------------
/**
  * Get the addresses of 4 of the 2x1 neighbor pixels 
  * for each element of the uniform CSPad matrix. 
  */ 
void
CSPadInterpolImageProducer::get_address_of_4_neighbors(unsigned ix, unsigned iy)
{
      ArrAddr addr00 = m_address_table_1[ix][iy];

      if ( areEqual(addr00, m_addr_empty) ) return;

      int q00 = addr00.quad;
      int s00 = addr00.sect;
      int r00 = addr00.row;
      int c00 = addr00.col;

      int r01 = r00;
      int c01 = c00;
      int r10 = r00;
      int c10 = c00;
      int r11 = r00;
      int c11 = c00;

      int rp1 = (r00<(int)m_nrows2x1-1) ? r00+1 : r00;
      int rm1 = (r00>0                ) ? r00-1 :   0;
      int cp1 = (c00<(int)m_ncols2x1-1) ? c00+1 : c00;
      int cm1 = (c00>0                ) ? c00-1 :   0;

      double x00_and_half = 0.5 + m_pix_coords_cspad -> getPixCoor_pix(XCOOR, q00, s00, r00, c00);
      double y00_and_half = 0.5 + m_pix_coords_cspad -> getPixCoor_pix(YCOOR, q00, s00, r00, c00);

      // Find the indexes for the direction in which the x and y coordinates increase.
      // Find X axis direction:
           if( m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, rp1, c00) > x00_and_half ) { r01 = rp1; r11 = rp1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, rm1, c00) > x00_and_half ) { r01 = rm1; r11 = rm1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, r00, cp1) > x00_and_half ) { c01 = cp1; c11 = cp1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, r00, cm1) > x00_and_half ) { c01 = cm1; c11 = cm1; }

      // Find Y axis direction:
           if( m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, rp1, c00) > y00_and_half ) { r10 = rp1; r11 = rp1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, rm1, c00) > y00_and_half ) { r10 = rm1; r11 = rm1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, r00, cp1) > y00_and_half ) { c10 = cp1; c11 = cp1; }
      else if( m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, r00, cm1) > y00_and_half ) { c10 = cm1; c11 = cm1; }

      m_address[ix][iy][0] = addr00;
      m_address[ix][iy][1] = (ArrAddr) {q00, s00, r01, c01};
      m_address[ix][iy][2] = (ArrAddr) {q00, s00, r10, c10};
      m_address[ix][iy][3] = (ArrAddr) {q00, s00, r11, c11};
}

//--------------------
/**
  * Get the weights of 4 of the 2x1 neighbor pixels 
  * for each element of the uniform CSPad matrix. 
  * Interpolation algorithm assumes the 4-node formula:
  *
  *           Term                   Weight
  * f(x,y) =  f00                    1 
  *        + (f10-f00)*x             x
  *        + (f01-f00)*y             y
  *        + (f11+f00-f10-f01)*x*y   x*y
  */
void
CSPadInterpolImageProducer::get_weight_of_4_neighbors(unsigned ix, unsigned iy)
{
  //ArrAddr addr00 = m_address_table_1[ix][iy];
     if ( areEqual(m_address_table_1[ix][iy], m_addr_empty) ) return;

     int q00 = m_address[ix][iy][0].quad;
     int s00 = m_address[ix][iy][0].sect;

     /*
     double x[4];
     double y[4];

     for (unsigned ip=0; ip<4; ip++){
       x[ip] = m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, m_address[ix][iy][ip].row, m_address[ix][iy][ip].col);
       y[ip] = m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, m_address[ix][iy][ip].row, m_address[ix][iy][ip].col);
     }

     double dx = ( x[1]-x[0] > 0) ? ( (double)ix - x[0] ) / ( x[1] - x[0]) : 0;
     double dy = ( y[2]-y[0] > 0) ? ( (double)iy - y[0] ) / ( y[2] - y[0]) : 0;
     */

     double x00 = m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, m_address[ix][iy][0].row, m_address[ix][iy][0].col);
     double y00 = m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, m_address[ix][iy][0].row, m_address[ix][iy][0].col);
     double x01 = m_pix_coords_cspad->getPixCoor_pix(XCOOR, q00, s00, m_address[ix][iy][1].row, m_address[ix][iy][1].col);
     double y10 = m_pix_coords_cspad->getPixCoor_pix(YCOOR, q00, s00, m_address[ix][iy][2].row, m_address[ix][iy][2].col);

     double dx = (x01-x00 > 0) ? ( (double)ix - x00 ) / (x01-x00) : 0;
     double dy = (y10-y00 > 0) ? ( (double)iy - y00 ) / (y10-y00) : 0;
     double dxdy = dx*dy;

     m_weight [ix][iy][0] = 1;
     m_weight [ix][iy][1] = dx; 
     m_weight [ix][iy][2] = dy; 
     m_weight [ix][iy][3] = dxdy; 

     /*
     cout << "TEST:"
          << " ix="  << ix
          << " iy="  << iy
          << " x00=" << x00
          << " x01=" << x01
          << " y00=" << y00
          << " y10=" << y10
          << endl;
     */
}

//--------------------
//--------------------
//--------------------
//--------------------
/**
  * Initialization of the CSPad array
  */ 
void
CSPadInterpolImageProducer::cspad_image_init()
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
/**
  * Fill the CSPad array with interpolation.
  */ 
void
CSPadInterpolImageProducer::cspad_image_interpolated_fill (ndarray<int16_t, 3> data[], QuadParameters* quadpars[], bool quadIsAvailable[])
{
    for (unsigned ix=0; ix<NX_CSPAD; ix++){
    for (unsigned iy=0; iy<NY_CSPAD; iy++){

      double sum_wf = 0;
      double sum_w  = 0;
      for (unsigned ip=0; ip<4; ip++){ // loop over 4 neighbour pixels

          ArrAddr &addr = m_address[ix][iy][ip];

          // Skip empty bins of the CSPad detector image
          if ( addr.quad == -1 ) continue;

          // Skip missing quads
          if ( !quadIsAvailable[addr.quad] ) continue;

          // Skip missing 2x1 data
	  bool  bitIsOn = (quadpars[addr.quad]->getRoiMask()) & (1<<addr.sect);
          if ( !bitIsOn ) continue; 

          const double f = (double)data[addr.quad][addr.sect][addr.col][addr.row];
	  const double w = (double)m_weight [ix][iy][ip];

          sum_wf += (w * f);
          sum_w  +=  w;

       } // ip
       m_arr_cspad_image[ix][iy] = (sum_w != 0) ? sum_wf / sum_w : 0;

    } // iy
    } // ix
}

//--------------------
/**
  * Save the CSPad image array in the text file.
  */ 
void
CSPadInterpolImageProducer::cspad_image_save_in_file(const std::string &filename)
{
  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&m_arr_cspad_image[0][0],NY_CSPAD,NX_CSPAD);
  img2d -> saveImageInFile(filename,0);
}

//--------------------
/**
  * Add the CSPad image array in the event.
  */ 
void
CSPadInterpolImageProducer::cspad_image_add_in_event(Event& evt, const std::string &keyname)
{
  shared_ptr< CSPadPixCoords::Image2D<double> > img2d( new CSPadPixCoords::Image2D<double>(&m_arr_cspad_image[0][0],NY_CSPAD,NX_CSPAD) );
  evt.put(img2d, m_actualSrc, keyname);
}

//--------------------

} // namespace CSPadPixCoords
