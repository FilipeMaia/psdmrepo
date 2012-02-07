//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MiniCSPadPixSpectra...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgPixSpectra/MiniCSPadPixSpectra.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
// #include "psddl_psana/acqiris.ddl.h"

#include "psddl_psana/cspad2x2.ddl.h"
#include "PSEvt/EventId.h"
#include "CSPadPixCoords/Image2D.h"


//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgPixSpectra;
PSANA_MODULE_FACTORY(MiniCSPadPixSpectra)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgPixSpectra {

//----------------
// Constructors --
//----------------
MiniCSPadPixSpectra::MiniCSPadPixSpectra (const std::string& name)
  : Module(name)
  , m_src()
  , m_key()
  , m_amin()
  , m_amax()
  , m_nbins()
  , m_arr_fname()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_src           = configStr("source", "DetInfo(:Cspad2x2)");
  m_key           = configStr("inputKey",  "");
  m_amin          = config   ("amin",      0.);
  m_amax          = config   ("amax",   1000.);
  m_nbins         = config   ("nbins",    100);
  m_arr_fname     = configStr("arr_fname", "mini_cspad_spectral_array.txt");
  m_maxEvents     = config   ("events", 1<<31U);
  m_filter        = config   ("filter", false);
}

//--------------
// Destructor --
//--------------
MiniCSPadPixSpectra::~MiniCSPadPixSpectra ()
{
}

/// Method which is called once at the beginning of the job
void 
MiniCSPadPixSpectra::beginJob(Event& evt, Env& env)
{
  this -> printInputPars();
  this -> arrayInit();
}

/// Method which is called at the beginning of the run
void 
MiniCSPadPixSpectra::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
MiniCSPadPixSpectra::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
MiniCSPadPixSpectra::event(Event& evt, Env& env)
{
  //// example of getting non-detector data from event
  //shared_ptr<PSEvt::EventId> eventId = evt.get();
  //if (eventId.get()) {
  //  // example of producing messages using MgsLog facility
  //  MsgLog(name(), info, "event ID: " << *eventId);
  //}
  
  //// tis is how to skip event (all downstream modules will not be called)
  //if (m_filter && m_count % 10 == 0) { skip(); return; }
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) { stop(); return; }

  if (   m_count<5 
     or (m_count<500 and m_count%100  == 0) 
     or                  m_count%1000 == 0  ) WithMsgLog(name(), info, log) { log << "event=" << m_count; }
  
  shared_ptr<Psana::CsPad2x2::ElementV1> mini1 = evt.get(m_src, m_key, &m_actualSrc);
  if (mini1.get()) {

      const Psana::CsPad2x2::ElementV1& el = *mini1;
      
      //const int16_t* data = el.data(); // depricated
      const ndarray<int16_t,3>& data_nda = el.data();
      const int16_t* data = &data_nda[0][0][0];

      const unsigned* dshape = data_nda.shape();
      int npix_mini1 = dshape[0] * dshape[1] * dshape[2];  

      if (npix_mini1 != m_npix_mini1) {
          MsgLog(name(), error, "Unexpected data shape: " << npix_mini1 << ", expected MiniElementV1 data size is " << m_npix_mini1);
          stop();
      }

      //WithMsgLog(name(), info, log) { 
      //  log << "CsPad2x2::ElementV1  dshape.size() = " << dshape.size() << "  shape = ";
      //for(uint i=0; i<dshape.size(); i++) log << dshape[i] << ", "; 
      //log << "    npix_mini1 = " << npix_mini1 << "\n";
      //

      this -> arrayFill (data);
  }

  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
MiniCSPadPixSpectra::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
MiniCSPadPixSpectra::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
MiniCSPadPixSpectra::endJob(Event& evt, Env& env)
{
  MsgLog(name(), info, "MiniCSPadPixSpectra::endJob");
  this -> saveArrayInFile();
  this -> saveShapeInFile();
  this -> arrayDelete();
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
MiniCSPadPixSpectra::arrayInit()
{
  m_factor = double(m_nbins) / (m_amax-m_amin);  // scale factor for histogramm index
  m_nbins1 = m_nbins - 1;

  int size = m_npix_mini1 * m_nbins;
  m_arr    = new int [size];
  for(int i=0; i<size; i++) m_arr[i] = 0;
}

//--------------------

void 
MiniCSPadPixSpectra::arrayDelete()
{
  delete [] m_arr;
}

//--------------------

void
MiniCSPadPixSpectra::arrayFill(const int16_t* data)
{
             for (uint32_t i=0; i<m_npix_mini1; i++) {
               double amp = (double)data[i];
               int iamp = this -> ampToIndex(amp);
	       //cout << "pix=" << i << " amp=" << amp << endl;  
               m_arr[i * m_nbins + iamp] ++; // incriment in spectral array
             }
}

//--------------------

void 
MiniCSPadPixSpectra::saveArrayInFile()
{ 
    MsgLog(name(), info, "Save the spectral array in file " << m_arr_fname);
    CSPadPixCoords::Image2D<int>* arr = new CSPadPixCoords::Image2D<int>(&m_arr[0], m_npix_mini1, m_nbins); 
    arr -> saveImageInFile(m_arr_fname,0);
}

//--------------------

void 
MiniCSPadPixSpectra::saveShapeInFile()
{ 
    m_arr_shape_fname = m_arr_fname + ".sha";
    MsgLog(name(), info, "Save the spectral array configuration in file " << m_arr_shape_fname);
    ofstream file; 
    file.open(m_arr_shape_fname.c_str(),ios_base::out);
    file << "NPIXELS  " << m_npix_mini1      << "\n";
    file << "NBINS    " << m_nbins           << "\n";
    file << "AMIN     " << m_amin            << "\n";
    file << "AMAX     " << m_amax            << "\n";
    file << "NEVENTS  " << m_count           << "\n";
    file << "ARRFNAME " << m_arr_fname       << "\n";
    file.close();
}

//--------------------

int  
MiniCSPadPixSpectra::ampToIndex(double amp)
{
    int ind = (int) (m_factor*(amp-m_amin));
    if( ind < 0       ) return 0;
    if( ind > m_nbins1) return m_nbins1;
    return ind;
}

//--------------------

void 
MiniCSPadPixSpectra::printInputPars()
{
  WithMsgLog(name(), info, log) { log 
        << "\n    Input parameters:"
      //<< "\n    m_src         " << m_src       
        << "\n    m_key         " << m_key    
        << "\n    m_maxEvents   " << m_maxEvents 
        << "\n    m_amin        " << m_amin      
        << "\n    m_amax        " << m_amax      
        << "\n    m_nbins       " << m_nbins     
        << "\n    m_arr_fname   " << m_arr_fname    
      //<< "\n    m_filter      " << m_filter
        << "\n";
      }
}

//--------------------

} // namespace ImgPixSpectra

