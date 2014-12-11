//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CSPadPixCoords/CSPadImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/EventId.h"
#include "PSCalib/CalibFileFinder.h"
#include "PSCalib/GeometryAccess.h"
#include "PSCalib/SegGeometryCspad2x1V1.h"
//#include "PSCalib/PixCoords2x1V2.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(CSPadImageProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {


//----------------
// Constructors --
//----------------

CSPadImageProducer::CSPadImageProducer (const std::string& name)
  : Module(name)
  , m_calibDir()
  , m_typeGroupName()
  , m_str_src()
  , m_inkey()
  , m_imgkey()
  , m_tiltIsApplied()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_calibDir      = configStr("calibDir",      ""); // if not provided default from env will be used
  m_typeGroupName = configStr("typeGroupName", "CsPad::CalibV1");
  m_str_src       = configStr("source",        "CxiDs1.0:Cspad.0");
  m_inkey         = configStr("key",           "");
  m_imgkey        = configStr("imgkey",        "image");
  m_fname_pixmap  = configStr("fname_pixmap",  "");
  m_fname_pixnum  = configStr("fname_pixnum",  "");
  m_tiltIsApplied = config   ("tiltIsApplied", true);
  m_print_bits    = config   ("print_bits",    0);

  m_source        = Source(m_str_src);
  m_count_msg= 0;

//checkTypeImplementation();
}


//--------------
// Destructor --
//--------------

CSPadImageProducer::~CSPadImageProducer ()
{
}

//--------------------

/// Print input parameters
void 
CSPadImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\ncalibDir      : "     << m_calibDir     
        << "\ntypeGroupName : "     << m_typeGroupName
        << "\nstr_src       : "     << m_str_src      
        << "\nsource        : "     << m_source      
        << "\nkey           : "     << m_inkey        
        << "\nimgkey        : "     << m_imgkey       
        << "\nfname_pixmap  : "     << m_fname_pixmap       
        << "\nfname_pixnum  : "     << m_fname_pixnum 
        << "\ntiltIsApplied : "     << m_tiltIsApplied
        << "\nprint_bits    : "     << m_print_bits
        << "\n";     
  }
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPadImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 16) printSizeOfTypes();
}

//--------------------

/// Method which is called at the beginning of the run
void 
CSPadImageProducer::beginRun(Event& evt, Env& env)
{
  m_count_cfg = 0; 
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
CSPadImageProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadImageProducer::event(Event& evt, Env& env)
{
  ++m_count;
  if( m_print_bits & 4 ) printTimeStamp(evt, m_count);


  if ( m_count_cfg==0 ) {
      getConfigPars(env);     // get m_src, incriments m_count_cfg

      if ( m_count_cfg==0 ) return; // skip event processing if configuration is missing

      getCalibPars(evt, env); // use m_src
      cspadImgActivePixelMask(env);
  }


  struct timespec start, stop;
  int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time

  procEvent(evt, env);

  if( m_print_bits & 4 ) {
    status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
    //cout << "  Time to produce cspad image is " 
    //     << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
    //     << " sec" << endl;
    MsgLog(name(), info, "Time to produce cspad image is " 
         << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
         << " sec");
  }
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
CSPadImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
CSPadImageProducer::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
CSPadImageProducer::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
CSPadImageProducer::getConfigPars(Env& env)
{
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV2>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV3>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV4>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV5>(env) ) return;

  m_count_msg++;
  if (m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "CsPad::ConfigV2-V5 is not available in this run, event:" << m_count << " for source:" << m_str_src);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src);
  }
}

//--------------------

bool
CSPadImageProducer::getGeometryPars(const std::string& calib_dir, const int runnum, const unsigned prbits)
{
  PSCalib::CalibFileFinder calibfinder(calib_dir, m_typeGroupName, prbits);
  const std::string calibtype("geometry");
  std::string fname = calibfinder.findCalibFile(m_src, calibtype, runnum);

  if( fname.empty() ) return false;
  if( m_print_bits & 2 ) MsgLog(name(), info, "Use \"geometry\" constants for run: " << runnum << " from file:\n" << fname);

  PSCalib::GeometryAccess geometry(fname, prbits);
  if( m_print_bits & 2 ) geometry.print_pixel_coords();

  const double* X;
  const double* Y;
  const double* Z;
  unsigned   size;
  geometry.get_pixel_coords(X,Y,Z,size);

  const double um_to_pix   = PSCalib::SegGeometryCspad2x1V1::UM_TO_PIX;
  const double pix_size_um = PSCalib::SegGeometryCspad2x1V1::PIX_SCALE_SIZE;

  // find minimal X,Y coordinates
  double Xmin = 1e6;
  double Ymin = 1e6;
  for(unsigned i=0; i<size; ++i) {
    if(X[i]<Xmin) Xmin=X[i];
    if(Y[i]<Ymin) Ymin=Y[i];
  }

  // apply coordinate offset and evaluate indexes
  m_coor_x_int = new uint32_t[size];
  m_coor_y_int = new uint32_t[size];

  for(unsigned i=0; i<size; ++i) {
    m_coor_x_int[i] = uint32_t( (X[i]-Xmin+0.5*pix_size_um)*um_to_pix );
    m_coor_y_int[i] = uint32_t( (Y[i]-Ymin+0.5*pix_size_um)*um_to_pix );
  }

  return true;
  //return false;
}

//--------------------

/// Method which is called at the beginning of the run
void 
CSPadImageProducer::getCalibPars(Event& evt, Env& env)
{
  std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  int runnum = getRunNumber(evt);
  unsigned prbits = (m_print_bits & 64) ? 0377 : 0;


  if( getGeometryPars(calib_dir, runnum, prbits) ) return;


  m_cspad_calibpar   = new PSCalib::CSPadCalibPars(calib_dir, m_typeGroupName, m_src, runnum, prbits);
  m_pix_coords_2x1   = new CSPadPixCoords::PixCoords2x1   ();
  m_pix_coords_quad  = new CSPadPixCoords::PixCoordsQuad  ( m_pix_coords_2x1,  m_cspad_calibpar, m_tiltIsApplied );
  m_pix_coords_cspad = new CSPadPixCoords::PixCoordsCSPad ( m_pix_coords_quad, m_cspad_calibpar, m_tiltIsApplied );

  m_coor_x_pix = m_pix_coords_cspad -> getPixCoorArrX_pix();
  m_coor_y_pix = m_pix_coords_cspad -> getPixCoorArrY_pix();
  m_coor_x_int = m_pix_coords_cspad -> getPixCoorArrX_int();
  m_coor_y_int = m_pix_coords_cspad -> getPixCoorArrY_int();

  if( m_print_bits & 2 ) m_cspad_calibpar  -> printCalibPars();
  //m_pix_coords_2x1  -> print_member_data();
  //m_pix_coords_quad -> print_member_data(); 
}

//--------------------

void 
CSPadImageProducer::getCSPadConfigFromData(Event& evt)
{
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) return;
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) return;

  m_count_msg++;
  if (m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "getCSPadConfigFromData(...): Psana::CsPad::DataV#,ElementV# for #=1,2 is not available in event:"
                            << m_count << " for source:" << m_str_src);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src);
  }
}

//--------------------

void 
CSPadImageProducer::procEvent(Event& evt, Env& env)
{  
  getCSPadConfigFromData(evt);

  // Check if the requested src and key are consistent with Psana::CsPad::DataV1, or V2
  if ( procCSPadDataForType  <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) return;
  if ( procCSPadDataForType  <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) return;

  // Check if the requested src and key are consistent with ndarray<T,3> of shape [N][185][388]
  if ( procCSPadNDArrForType <float>    (evt) ) return;
  if ( procCSPadNDArrForType <double>   (evt) ) return;
  if ( procCSPadNDArrForType <int>      (evt) ) return;
  if ( procCSPadNDArrForType <int16_t>  (evt) ) return;
  if ( procCSPadNDArrForType <uint16_t> (evt) ) return;

  m_count_msg++;
  if (m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "procEvent(...): Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in event:"
                             << m_count << " for source:" << m_str_src << " key:" << m_inkey);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP PRINTING WARNINGS for source:" << m_str_src << " key:" << m_inkey);
  }
}

//--------------------

void 
CSPadImageProducer::checkTypeImplementation()
{  
  if ( m_outtype == "float"   ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double"  ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"     ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"   ) { m_dtype = INT16;  return; } 
  if ( m_outtype == "int16_t" ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------
void 
CSPadImageProducer::cspadImgActivePixelMask(Env& env) 
{
        const unsigned shape[] = {NY_CSPAD,NX_CSPAD};
        ndarray<pixmap_cspad_t,2> pixmap_2da(shape);
        ndarray<pixnum_cspad_t,2> pixnum_2da(shape);
        std::fill(pixmap_2da.begin(), pixmap_2da.end(), pixmap_cspad_t(0));    
        std::fill(pixnum_2da.begin(), pixnum_2da.end(), pixnum_cspad_t(-1));    
        //std::fill_n(img_nda.data(), int(IMG_SIZE), mask_cspad_t(0));    

	int ix=0; int iy=0;
        for(uint32_t pix=0; pix<ARR_SIZE; pix++)
        {
               ix = m_coor_x_int[pix];
               iy = m_coor_y_int[pix];
	       pixmap_2da[ix][iy] = 1;
	       pixnum_2da[ix][iy] = pix;
	}

        save2DArrayInEnv<pixmap_cspad_t>(env, m_src, pixmap_2da);
        save2DArrayInEnv<pixnum_cspad_t>(env, m_src, pixnum_2da);

        if( ! m_fname_pixmap.empty() ) {
          const std::string msg = "Save active pixel map in file: " + m_fname_pixmap;
          MsgLog(name(), info, msg );
	  save2DArrayInFile<pixmap_cspad_t>(m_fname_pixmap, pixmap_2da, m_print_bits&32);
	}

        if( ! m_fname_pixnum.empty() ) {
          const std::string msg = "Save enumerated pixel map in file: " + m_fname_pixnum;
          MsgLog(name(), info, msg );
	  save2DArrayInFile<pixnum_cspad_t>(m_fname_pixnum, pixnum_2da, m_print_bits&32);
	}
}

//--------------------
 
} // namespace CSPadPixCoords
