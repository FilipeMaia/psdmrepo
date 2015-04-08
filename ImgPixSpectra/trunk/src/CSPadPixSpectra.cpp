//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadPixSpectra...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgPixSpectra/CSPadPixSpectra.h"

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
// #include "psddl_psana/cspad.ddl.h" // moved to header
#include "PSEvt/EventId.h"
#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgPixSpectra;
PSANA_MODULE_FACTORY(CSPadPixSpectra)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgPixSpectra {

//----------------
// Constructors --
//----------------
CSPadPixSpectra::CSPadPixSpectra (const std::string& name)
  : Module(name)
  , m_src()
  , m_key()
  , m_maxEvents()
  , m_amin()
  , m_amax()
  , m_nbins()
  , m_arr_fname()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_src           = configStr("source", "CxiDs1.0:Cspad.0");
  m_key           = configStr("inputKey",   "");
  m_maxEvents     = config   ("events", 1<<31U);
  m_amin          = config   ("amin",       0.);
  m_amax          = config   ("amax",    1000.);
  m_nbins         = config   ("nbins",     100);
  m_arr_fname     = configStr("arr_fname", "cspad_spectral_array.txt");
  m_filter        = config   ("filter",  false); 
}

//--------------
// Destructor --
//--------------
CSPadPixSpectra::~CSPadPixSpectra ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadPixSpectra::beginJob(Event& evt, Env& env)
{
  this -> printInputPars();
  this -> getQuadConfigPars(env);
  this -> printQuadConfigPars();
  this -> arrayInit();
}

/// Method which is called at the beginning of the run
void 
CSPadPixSpectra::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadPixSpectra::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadPixSpectra::event(Event& evt, Env& env)
{
  // example of getting non-detector data from event
  //shared_ptr<PSEvt::EventId> eventId = evt.get();
  //if (eventId.get()) {
  //  // example of producing messages using MgsLog facility
  //  MsgLog(name(), info, "event " << m_count << " ID: " << *eventId);
  //}
  
  // this is how to skip event (all downstream modules will not be called)
  //if (m_filter && m_count % 10 == 0) skip();
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) {stop(); return;}

  // getting detector data from event
  //shared_ptr<Psana::CsPad::DataV2> data = evt.get(m_src, "", &m_actualSrc); // get m_actualSrc here
  shared_ptr<CSPadDataType> data = evt.get(m_src, m_key, &m_actualSrc); // get m_actualSrc here

  if (   m_count<5 
     or (m_count<500 and m_count%100  == 0) 
     or                  m_count%1000 == 0  ) WithMsgLog(name(), info, log) { log << "event=" << m_count; }

  if (data.get()) {
    this -> loopOverQuads(data);
  }
  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadPixSpectra::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadPixSpectra::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadPixSpectra::endJob(Event& evt, Env& env)
{
  this -> saveArrayInFile();
  this -> saveShapeInFile();
  this -> arrayDelete();
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
CSPadPixSpectra::arrayInit()
{
  m_factor = double(m_nbins) / (m_amax-m_amin);  // scale factor for histogramm index
  m_nbins1 = m_nbins - 1;

  int size = m_sizeOfCSPadArr * m_nbins;
  m_arr    = new int [size];
  for(int i=0; i<size; i++) m_arr[i] = 0;

  // m_arr2d = make_ndarray(m_arr, m_sizeOfCSPadArr, m_nbins);
}

//--------------------

void 
CSPadPixSpectra::arrayDelete()
{
  delete [] m_arr;
}

//--------------------

void 
CSPadPixSpectra::loopOverQuads(shared_ptr<CSPadDataType> data)
{
    m_pixel_ind = 0;
    int nQuads = data -> quads_shape()[0];

    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data->quads(q);

        int quad                           = el.quad() ;
        const ndarray<const int16_t,3>& data_nda = el.data();
        const int16_t* data = &data_nda[0][0][0];

          //cout << "     q = " << q << " quad =" << quad << endl;
        this -> arrayFill (quad, data, m_roiMask[q]);
    }
}

//--------------------

void
CSPadPixSpectra::arrayFill(int quad, const int16_t* data, uint32_t roiMask)
{
        for(uint32_t sect=0; sect < m_n2x1; sect++)
        {
             bool bitIsOn = roiMask & (1<<sect);
             if( !bitIsOn ) { m_pixel_ind += m_sizeOf2x1Arr; continue; }
 
             const int16_t *data2x1 = &data[sect * m_sizeOf2x1Arr];

             //cout  << "  add section " << sect << endl;            
 
             //for (uint32_t c=0; c<m_ncols2x1; c++) {
             //for (uint32_t r=0; r<m_nrows2x1; r++) {
	     //  double amp = (double)data2x1[c*m_nrows2x1+r];

             for (uint32_t i=0; i<m_sizeOf2x1Arr; i++) {
	       double amp = (double)data2x1[i];
	       int iamp = this -> ampToIndex(amp);

	       int ipix = m_pixel_ind ++;
               m_arr[ipix * m_nbins + iamp] ++; // incriment in spectral array
             }
        }
}

//--------------------

int  
CSPadPixSpectra::ampToIndex(double amp)
{
    int ind = (int) (m_factor*(amp-m_amin));
    if( ind < 0       ) return 0;
    if( ind > m_nbins1) return m_nbins1;
    return ind;
}

//--------------------

void 
CSPadPixSpectra::getQuadConfigPars(Env& env)
{
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV2>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV3>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV4>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV5>(env) ) return;

  MsgLog(name(), warning, "CsPad::ConfigV2 - V5 is not available in this run.");
}

//--------------------

void 
CSPadPixSpectra::printInputPars()
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

void 
CSPadPixSpectra::printQuadConfigPars()
{
  WithMsgLog(name(), info, log) { log 
        << "\n    CSPad configuration parameters:"
        << "\n    m_nquads             = " << m_nquads
        << "\n    m_n2x1 in quad       = " << m_n2x1
        << "\n    m_ncols2x1           = " << m_ncols2x1
        << "\n    m_nrows2x1           = " << m_nrows2x1
        << "\n    m_sizeOf2x1Arr       = " << m_sizeOf2x1Arr
        << "\n    m_sizeOfQuadArr      = " << m_sizeOfQuadArr
        << "\n    m_sizeOfCSPadArr     = " << m_sizeOfCSPadArr;

    for (uint32_t q = 0; q < m_nquads; ++ q) { log
        << "\nq = " << q
        << "  m_roiMask[q] = "         << m_roiMask[q]
        << "  m_numAsicsStored[q] = "  << m_numAsicsStored[q];
    }
    log << "\n"; 
  }
}

//--------------------

void 
CSPadPixSpectra::saveArrayInFile()
{ 
    MsgLog(name(), info, "Save the spectral array in file " << m_arr_fname);
    CSPadPixCoords::Image2D<int>* arr = new CSPadPixCoords::Image2D<int>(&m_arr[0], m_sizeOfCSPadArr, m_nbins); 
    arr -> saveImageInFile(m_arr_fname,0);
}

//--------------------

void 
CSPadPixSpectra::saveShapeInFile()
{ 
    m_arr_shape_fname = m_arr_fname + ".sha";
    MsgLog(name(), info, "Save the spectral array configuration in file " << m_arr_shape_fname);
    std::ofstream file; 
    file.open(m_arr_shape_fname.c_str(), std::ios_base::out);
    file << "NPIXELS  " << m_sizeOfCSPadArr  << "\n";
    file << "NBINS    " << m_nbins           << "\n";
    file << "AMIN     " << m_amin            << "\n";
    file << "AMAX     " << m_amax            << "\n";
    file << "NEVENTS  " << m_count           << "\n";
    file << "ARRFNAME " << m_arr_fname       << "\n";
    file.close();
}

//--------------------
//--------------------

} // namespace ImgPixSpectra
