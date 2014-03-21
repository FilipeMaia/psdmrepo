//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSpectraProc...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgSpectraProc.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <sstream> // for stringstream
#include <iomanip> // for setw, setfill

using namespace ImgAlgos;

PSANA_MODULE_FACTORY(ImgSpectraProc)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgSpectraProc::ImgSpectraProc (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src       = configSrc("source", "DetInfo(:Opal1000)");
  m_key_in        = configStr("key_in",                "img");
  m_print_bits    = config   ("print_bits",               0 );
}

//--------------------
/// Print input parameters
void 
ImgSpectraProc::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\nkey_in       : "     << m_key_in      
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------

//--------------
// Destructor --
//--------------

ImgSpectraProc::~ImgSpectraProc ()
{
}

//--------------------
//--------------------

/// Method which is called once at the beginning of the job
void 
ImgSpectraProc::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called at the beginning of the run
void 
ImgSpectraProc::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
ImgSpectraProc::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgSpectraProc::event(Event& evt, Env& env)
{
  ++m_count;
  getSpectra(evt, m_print_bits & 4);
  if( m_print_bits & 8 ) printSpectra(evt);
  if( m_print_bits & 2 ) printEventRecord(evt, " done...");
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
ImgSpectraProc::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
ImgSpectraProc::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
ImgSpectraProc::endJob(Event& evt, Env& env)
{
}

//--------------------
//--------------------

void 
ImgSpectraProc::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
                     << comment.c_str() 
  );
}

//--------------------

void 
ImgSpectraProc::printSpectra(Event& evt)
{
  unsigned dc = 100;
  MsgLog( name(), info, "Image spectra for run=" << stringRunNumber(evt) << " Evt=" << stringFromUint(m_count) );
  std::cout <<   "Column:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << c;
  std::cout << "\nSignal:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(0) << std::fixed << m_data[c];
  std::cout << "\nRefer.:"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(0) << std::fixed << m_data[c+m_cols];
  std::cout << "\nDiff. :"; for( unsigned c=0; c<m_cols; c+=dc ) std::cout << std::setw(8) << std::setprecision(3) << std::fixed << m_data[c+m_cols*2];
  std::cout << "\n";
}

//--------------------

void 
ImgSpectraProc::getSpectra(Event& evt, bool print_msg)
{
  shared_ptr< ndarray<const double,2> > sp = evt.get(m_str_src, m_key_in, &m_src);
  if (sp.get()) {
        m_data = sp->data(); 
	m_rows = sp->shape()[0];
	m_cols = sp->shape()[1];
	if(print_msg) MsgLog( name(), info, "Spectral array shape =" << m_rows << ", " << m_cols); 
  }
  else
  {
    MsgLog( name(), info, "Spectral array ndarray<double,2> is not available for source: " << m_str_src << " and key: " <<  m_key_in ); 
  }
}

//--------------------
//--------------------

} // namespace ImgAlgos
