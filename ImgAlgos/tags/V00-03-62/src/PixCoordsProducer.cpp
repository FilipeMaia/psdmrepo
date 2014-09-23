//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/PixCoordsProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream> // for cout, puts etc.
#include <sstream> // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h" // 
#include "PSCalib/CalibFileFinder.h"
//#include "PSEvt/EventId.h"

#include "psddl_psana/cspad.ddl.h"    // for Psana::CsPad::ConfigV#
#include "psddl_psana/cspad2x2.ddl.h" // for Psana::CsPad2x2::ConfigV#
#include "psddl_psana/epix.ddl.h"     // for Psana::Epix::ConfigV1

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(PixCoordsProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
PixCoordsProducer::PixCoordsProducer (const std::string& name)
  : Module(name)
  , m_src()
  , m_key_out_x() 
  , m_key_out_y() 
  , m_key_out_z() 
  , m_key_out_a() 
  , m_key_out_ix() 
  , m_key_out_iy() 
  , m_key_fname() 
  , m_print_bits(0)
  , m_count_run(0)
  , m_count_event(0)
  , m_count_calibcycle(0)
  , m_count_warnings(0)
  , m_count_cfg(0)
  , m_count_clb(0)
  , m_fname(std::string())
{
  m_str_src    = configSrc("source",    "DetInfo(:Cspad)");
  m_group      = configStr("group",     ""); // is set automaticly from source
  m_key_out_x  = configStr("key_out_x", "x-pix-coords");
  m_key_out_y  = configStr("key_out_y", "y-pix-coords");
  m_key_out_z  = configStr("key_out_z", "z-pix-coords");
  m_key_out_a  = configStr("key_out_area", "");
  m_key_out_ix = configStr("key_out_ix",   "");
  m_key_out_iy = configStr("key_out_iy",   "");
  m_key_fname  = configStr("key_fname",    "geometry-calib");
  m_print_bits = config   ("print_bits", 0);
}

//--------------------

void 
PixCoordsProducer::printInputParameters()
{
  MsgLog(name(), info, "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n group             : " << m_group
        << "\n key_out_x         : " << m_key_out_x
        << "\n key_out_y         : " << m_key_out_y
        << "\n key_out_z         : " << m_key_out_z
        << "\n key_out_a         : " << m_key_out_a
        << "\n key_out_ix        : " << m_key_out_ix
        << "\n key_out_iy        : " << m_key_out_iy
        << "\n key_fname         : " << m_key_fname
        << "\n print_bits        : " << m_print_bits
        << "\n";     
	 );
}

//--------------
// Destructor --
//--------------
PixCoordsProducer::~PixCoordsProducer ()
{
}

//--------------------
/// Method which is called once at the beginning of the job
void 
PixCoordsProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
PixCoordsProducer::beginRun(Event& evt, Env& env)
{
  ++ m_count_run;
  //cout << "m_count_run: " << m_count_run << "\n";

  m_count_clb = 0;
  checkCalibPars(evt, env);
}

/// Method which is called at the beginning of the calibration cycle
void 
PixCoordsProducer::beginCalibCycle(Event& evt, Env& env)
{
  ++ m_count_calibcycle;
  //cout << "m_count_calibcycle: " << m_count_calibcycle << "\n";
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
PixCoordsProducer::event(Event& evt, Env& env)
{
  ++ m_count_event;
  cout << "m_count_event: " << m_count_event << "\n";

  checkCalibPars(evt, env);

  //if( m_count_clb ) savePixCoordsInEvent(evt);
  if( m_count_clb ) savePixCoordsInCalibStore(env);
}
  
/// Method which is called at the end of the calibration cycle
void 
PixCoordsProducer::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
PixCoordsProducer::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
PixCoordsProducer::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
PixCoordsProducer::checkCalibPars(Event& evt, Env& env)
{
  if( m_count_clb ) return; // reset expernally in beginRun

  if( ! getConfigPars(env) ) return;
  if( ! getCalibPars(evt, env) ) return;

  ++ m_count_clb;
}

//--------------------
// This method is used to set Pds::Src m_src which is needed for saving info in the event store;

bool
PixCoordsProducer::getConfigPars(Env& env)
{
  m_count_cfg = 0;

  // check for CSPAD
  if ( getConfigParsForType<Psana::CsPad::ConfigV2>(env) ) { m_config_vers = "CsPad::ConfigV2"; return true; }
  if ( getConfigParsForType<Psana::CsPad::ConfigV3>(env) ) { m_config_vers = "CsPad::ConfigV3"; return true; }
  if ( getConfigParsForType<Psana::CsPad::ConfigV4>(env) ) { m_config_vers = "CsPad::ConfigV4"; return true; }
  if ( getConfigParsForType<Psana::CsPad::ConfigV5>(env) ) { m_config_vers = "CsPad::ConfigV5"; return true; }

  // check for CSPAD2x2
  if ( getConfigParsForType<Psana::CsPad2x2::ConfigV1>(env) ) { m_config_vers = "CsPad2x2::ConfigV1"; return true; }
  if ( getConfigParsForType<Psana::CsPad2x2::ConfigV2>(env) ) { m_config_vers = "CsPad2x2::ConfigV2"; return true; }

  // check for ePix
  if ( getConfigParsForType<Psana::Epix::ConfigV1>(env) ) { m_config_vers = "Epix::ConfigV1"; return true; }

  m_count_warnings++;
  if (m_count_warnings < 20) MsgLog(name(), warning, "CsPad::ConfigV2-5, CsPad2x2::ConfigV1-2, Epix::ConfigV1 is not available in this event...")
  if (m_count_warnings ==20) MsgLog(name(), warning, "STOP PRINT WARNINGS !!!")
  return false;
}

//--------------------

bool 
PixCoordsProducer::getCalibPars(Event& evt, Env& env)
{
  std::string calib_dir = env.calibDir();
  std::string group = (m_group.empty()) ? calibGroupForSource(m_str_src) : m_group; // for ex: "PNCCD::CalibV1";
  unsigned runnum = getRunNumber(evt);
  PSCalib::CalibFileFinder cff(calib_dir, group, 0);
  std::string fname = cff.findCalibFile(m_src, "geometry", runnum);

  if( m_print_bits &  2 ) {
     std::stringstream ss;
     ss << "Calibration directory: " << calib_dir   
        << "\n  source     : " << m_str_src
        << "\n  group      : " << group
        << "\n  run number : " << runnum 
        << "\n  calib file : " << fname; 
     MsgLog(name(), info, ss.str());
  }

  if( fname.empty() ) return false;

  if(fname == m_fname) return true; // if the file name has not changed for new run
  m_fname = fname;

  m_geometry = new PSCalib::GeometryAccess(fname, 0);
  if(! m_key_out_x.empty()
  or ! m_key_out_y.empty()
  or ! m_key_out_z.empty())  m_geometry -> get_pixel_coords(m_pixX, m_pixY, m_pixZ, m_size);

  if(! m_key_out_a .empty()) m_geometry -> get_pixel_areas(m_pixA, m_size_a);

  if(! m_key_out_ix.empty()
  or ! m_key_out_iy.empty()) m_geometry -> get_pixel_coord_indexes(m_pixIX, m_pixIY, m_size_ind);

  if( m_print_bits &  4 ) m_geometry -> print_list_of_geos();
  if( m_print_bits &  8 ) m_geometry -> print_list_of_geos_children();
  if( m_print_bits & 16 ) m_geometry -> print_comments_from_dict();
  if( m_print_bits & 32 ){m_geometry -> print_pixel_coords();
    cout << "X: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << m_pixX[i] << ", "; cout << "...\n"; 
  }

  if(! m_key_out_x .empty()) m_ndaX = make_ndarray(m_pixX,  m_size);
  if(! m_key_out_y .empty()) m_ndaY = make_ndarray(m_pixY,  m_size);
  if(! m_key_out_z .empty()) m_ndaZ = make_ndarray(m_pixZ,  m_size);
  if(! m_key_out_a .empty()) m_ndaA = make_ndarray(m_pixA,  m_size_a); 
  if(! m_key_out_ix.empty()) m_ndaIX= make_ndarray(m_pixIX, m_size_ind); 
  if(! m_key_out_iy.empty()) m_ndaIY= make_ndarray(m_pixIY, m_size_ind); 
  if(! m_key_fname .empty()) m_ndafn= make_ndarray(reinterpret_cast<const uint8_t*>(m_fname.c_str()), m_fname.size()); 

  return true;
}

//--------------------

void 
PixCoordsProducer::savePixCoordsInEvent(Event& evt)
{
  if(! m_key_out_x .empty()) save1DArrayInEvent<PixCoordsProducer::coord_t>      (evt, m_src, m_key_out_x,  m_ndaX);
  if(! m_key_out_y .empty()) save1DArrayInEvent<PixCoordsProducer::coord_t>      (evt, m_src, m_key_out_y,  m_ndaY);
  if(! m_key_out_z .empty()) save1DArrayInEvent<PixCoordsProducer::coord_t>      (evt, m_src, m_key_out_z,  m_ndaZ);
  if(! m_key_out_a .empty()) save1DArrayInEvent<PixCoordsProducer::area_t>       (evt, m_src, m_key_out_a,  m_ndaA);
  if(! m_key_out_ix.empty()) save1DArrayInEvent<PixCoordsProducer::coord_index_t>(evt, m_src, m_key_out_ix, m_ndaIX);
  if(! m_key_out_iy.empty()) save1DArrayInEvent<PixCoordsProducer::coord_index_t>(evt, m_src, m_key_out_iy, m_ndaIY);
  if(! m_key_fname .empty()) save1DArrayInEvent<uint8_t>                         (evt, m_src, m_key_fname,  m_ndafn);
}

//--------------------

void 
PixCoordsProducer::savePixCoordsInCalibStore(Env& env)
{
  // For now there is no key parameter in evr.calibStore().put(...) method.
  // Data can be only distinguished by type. So, there is no chance to save 3 ndarray<double,1>... 

  if(! m_key_out_x .empty()) save1DArrayInCalibStore<PixCoordsProducer::coord_t>      (env, m_src, "x-pix-coords", m_ndaX);
  if(! m_key_out_y .empty()) save1DArrayInCalibStore<PixCoordsProducer::coord_t>      (env, m_src, "y-pix-coords", m_ndaY);
  if(! m_key_out_z .empty()) save1DArrayInCalibStore<PixCoordsProducer::coord_t>      (env, m_src, "z-pix-coords", m_ndaZ);
  if(! m_key_out_a .empty()) save1DArrayInCalibStore<PixCoordsProducer::area_t>       (env, m_src, m_key_out_a,  m_ndaA);
  if(! m_key_out_ix.empty()) save1DArrayInCalibStore<PixCoordsProducer::coord_index_t>(env, m_src, m_key_out_ix, m_ndaIX);
  if(! m_key_out_iy.empty()) save1DArrayInCalibStore<PixCoordsProducer::coord_index_t>(env, m_src, m_key_out_iy, m_ndaIY);
  if(! m_key_fname .empty()) save1DArrayInCalibStore<uint8_t>                         (env, m_src, m_key_fname,  m_ndafn);
}

//--------------------

} // namespace ImgAlgos
