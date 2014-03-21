//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSpectra...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgSpectra.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "PSEvt/EventId.h"
//#include "CSPadPixCoords/Image2D.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgSpectra)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgSpectra::ImgSpectra (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_sig_rowc()
  , m_ref_rowc()
  , m_sig_tilt()
  , m_ref_tilt()
  , m_sig_width()
  , m_ref_width()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configSrc("source", "DetInfo(:Opal1000)");
  m_key_in        = configStr("key_in",                "img");
  m_key_out       = configStr("key_out",           "spectra");
  m_sig_rowc      = config   ("sig_band_rowc",          510.);
  m_ref_rowc      = config   ("ref_band_rowc",          550.);
  m_sig_tilt      = config   ("sig_band_tilt",            0.);
  m_ref_tilt      = config   ("ref_band_tilt",            0.);
  m_sig_width     = config   ("sig_band_width",          10 );
  m_ref_width     = config   ("ref_band_width",          10 );
  m_print_bits    = config   ("print_bits",               0 );
}

//--------------------
/// Print input parameters
void 
ImgSpectra::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\nkey_in       : "     << m_key_in      
        << "\nkey_out      : "     << m_key_out
        << "\nsig_rowc     : "     << m_sig_rowc 
        << "\nref_rowc     : "     << m_ref_rowc 
        << "\nsig_tilt     : "     << m_sig_tilt 
        << "\nref_tilt     : "     << m_ref_tilt 
        << "\nsig_width    : "     << m_sig_width
        << "\nref_width    : "     << m_ref_width
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------

//--------------
// Destructor --
//--------------

ImgSpectra::~ImgSpectra ()
{
}

//--------------------
//--------------------

/// Method which is called once at the beginning of the job
void 
ImgSpectra::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgSpectra::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgSpectra::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgSpectra::event(Event& evt, Env& env)
{
  ++m_count;

  procEvent(evt);
  if( m_print_bits & 8 ) printSpectra(evt);
  if( m_print_bits & 2 ) printEventRecord(evt, " is processed.");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgSpectra::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgSpectra::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgSpectra::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------

void 
ImgSpectra::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
                     << comment.c_str() 
  );
}

//--------------------

void 
ImgSpectra::printSpectra(Event& evt)
{
  unsigned dc = 100;
  MsgLog( name(), info, "Image spectra for run=" << stringRunNumber(evt) << " Evt=" << stringFromUint(m_count) );
  std::cout <<   "Column:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << c;
  std::cout << "\nSignal:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(0) << std::fixed << m_data[0][c];
  std::cout << "\nRefer.:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(0) << std::fixed << m_data[1][c];
  std::cout << "\nDiff. :"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(3) << std::fixed << m_data[2][c];
  std::cout << "\n";
}

//--------------------

void
ImgSpectra::difSpectrum()
{
      for( unsigned c=0; c<m_cols; c++ ) {
	double sum = m_data[0][c] + m_data[1][c];
	m_data[2][c] = (sum > 0) ? 2*(m_data[0][c] - m_data[1][c]) / sum : 0;
      }
}

//--------------------

void 
ImgSpectra::procEvent(Event& evt)
{
  shared_ptr< ndarray<const double,2> > sp_img = evt.get(m_str_src, m_key_in, &m_src);
  if (sp_img) {
    retrieveSpectra<double> (*sp_img, m_print_bits & 4);
    save2DArrayInEvent<double> (evt, m_src, m_key_out, m_data);
  }

  shared_ptr< ndarray<const uint16_t,2> > sp_img_u16 = evt.get(m_str_src, m_key_in, &m_src);
  if (sp_img_u16) {
    retrieveSpectra<uint16_t> (*sp_img_u16, m_print_bits & 4);
    save2DArrayInEvent<double> (evt, m_src, m_key_out, m_data);
  }
}

//--------------------
//--------------------

} // namespace ImgAlgos
