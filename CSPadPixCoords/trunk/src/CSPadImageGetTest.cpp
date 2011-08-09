//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadImageGetTest...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPadImageGetTest.h"

//-----------------
// C/C++ Headers --
//-----------------
// #include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "CSPadPixCoords/Image2D.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>
#include <iomanip> // for setw, setfill
#include <sstream> // for streamstring

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(CSPadImageGetTest)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPadImageGetTest::CSPadImageGetTest (const std::string& name)
  : Module(name)
  , m_source()
  , m_src()
  , m_maxEvents()
  , m_eventSave()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_source        = configStr("source",        "CxiDs1.0:Cspad.0");
  m_maxEvents     = config   ("events",        10000000);
  m_eventSave     = config   ("eventSave",     32U);
  m_filter        = config   ("filter",        false);
  m_src           = m_source;
}


//--------------
// Destructor --
//--------------

CSPadImageGetTest::~CSPadImageGetTest ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPadImageGetTest::beginJob(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the run
void 
CSPadImageGetTest::beginRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
CSPadImageGetTest::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadImageGetTest::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  ++m_count;
  if (m_count >= m_maxEvents) stop();
  cout << "CSPadImageGetTest::event(): Event: " << m_count << endl;

  if (m_count == m_eventSave) this -> saveImageInFile(evt);
}

//--------------------
  
/// Method which is called at the end of the calibration cycle
void 
CSPadImageGetTest::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
CSPadImageGetTest::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
CSPadImageGetTest::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
CSPadImageGetTest::saveImageInFile(Event& evt)
{
  shared_ptr< CSPadPixCoords::Image2D<double> > img2d = evt.get(m_src, "CSPad:Image", &m_actualSrc);

  if (img2d.get()) {

    cout << "CSPadImageGetTest::event(...): Get image from event and save it in file" << endl; 
    stringstream ssEvNum; ssEvNum << setw(6) << setfill('0') << m_count;
    string fname = "cspad_image_get_test_ev" + ssEvNum.str() + ".txt";
 
    img2d -> saveImageInFile(fname,0);
  } // if (img2d.get())
}

//--------------------

} // namespace CSPadPixCoords
