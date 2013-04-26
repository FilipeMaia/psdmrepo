//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgIntForBins...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgIntForBins.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgIntForBins)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgIntForBins::ImgIntForBins (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_fname_map_bins()
  , m_fname_int_bins()
  , m_nbins()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src        = configSrc("source",   "DetInfo(:Camera)");
  m_key_in         = configStr("key_in",                   "");
  m_key_out        = configStr("key_out",        "int_binned");
  m_fname_map_bins = configStr("fname_map_bins",           "");
  m_fname_int_bins = configStr("fname_int_bins",           "");
  m_nbins          = config   ("number_of_bins",          10 );
  m_print_bits     = config   ("print_bits",               0 );

  m_do_binning = (m_fname_map_bins.empty() or m_fname_int_bins.empty()) ? false : true;
}

//--------------------

void 
ImgIntForBins::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n m_key_in          : " << m_key_in      
        << "\n m_key_out         : " << m_key_out      
        << "\n m_fname_map_bins  : " << m_fname_map_bins
        << "\n m_fname_int_bins  : " << m_fname_int_bins
        << "\n m_nbins           : " << m_nbins
        << "\n m_do_binning      : " << m_do_binning
        << "\n m_print_bits      : " << m_print_bits
	<< "\n Output data type  : " << typeid(data_out_t).name() << " of size " << sizeof(data_out_t)
        << "\n";     

    printSizeOfTypes();
  }
}

//--------------------


//--------------
// Destructor --
//--------------
ImgIntForBins::~ImgIntForBins ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgIntForBins::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();

}

/// Method which is called at the beginning of the run
void 
ImgIntForBins::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgIntForBins::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgIntForBins::event(Event& evt, Env& env)
{
  if(!m_do_binning) return;
  if(!m_count) init(evt, env);
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt, env);
  // saveImageInEvent(evt); -> moved to procEventForType
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgIntForBins::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgIntForBins::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgIntForBins::endJob(Event& evt, Env& env)
{
  if(!m_do_binning) return;
  p_out.close();
  if( m_print_bits & 32 ) MsgLog( name(), info, "Intensity in bins vs evts is saved in the file: " << m_fname_int_bins.c_str());  
}

//--------------------

void 
ImgIntForBins::init(Event& evt, Env& env)
{
    defineImageShape(evt, m_str_src, m_key_in, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;

    if( m_do_binning ) m_map_bins = new ImgAlgos::ImgParametersV1(m_fname_map_bins);
    else               m_map_bins = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    ImgParametersV1::pars_t* map_bins_inds = m_map_bins->data();
    if( m_do_binning && m_print_bits & 4 ) m_map_bins -> print("Map of bin indexes:");

    m_inds = new unsigned [m_size];
    for( unsigned i=0; i<m_size; i++ ) m_inds[i] = (unsigned) map_bins_inds[i];

    m_intens_ave = new data_out_t [m_nbins];
    m_sum_intens = new data_out_t [m_nbins];
    m_sum_stat   = new unsigned   [m_nbins];    

    p_out.open(m_fname_int_bins.c_str());
    if( m_print_bits & 32 ) MsgLog( name(), info, "Open output file: " <<  m_fname_int_bins.c_str());  
}

//--------------------

void 
ImgIntForBins::procEvent(Event& evt, Env& env)
{
  if ( procEventForType<uint16_t> (evt) ) return;
  if ( procEventForType<int>      (evt) ) return;
  if ( procEventForType<float>    (evt) ) return;
  if ( procEventForType<uint8_t>  (evt) ) return;
  if ( procEventForType<double>   (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key_in);
}

//--------------------
 
std::string  
ImgIntForBins::strRecord()
{
  std::stringstream ss;
  ss << right << std::setw(8) << m_count << "  " << fixed << std::setprecision(3);
  for(unsigned bin=0; bin<m_nbins; bin++) ss << m_intens_ave[bin] << " "; ss << "\n";
  return ss.str();
}

//--------------------

void 
ImgIntForBins::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------
//--------------------
} // namespace ImgAlgos
