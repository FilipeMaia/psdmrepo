//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadTest...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/CSPadTest.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
// to work with detector data include corresponding 
// header from psddl_psana package
// #include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/cspad.ddl.h"

#include "CSPadImage/Image2D.h"
#include "CSPadImage/ImageCSPad2x1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <string.h>
//#include <memory>
//using namespace std;

// This declares this class as psana module
using namespace CSPadImage;
PSANA_MODULE_FACTORY(CSPadTest)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadImage {

//----------------
// Constructors --
//----------------
CSPadTest::CSPadTest (const std::string& name)
  : Module(name)
  , m_src()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_src       = configStr("source", "DetInfo(:Acqiris)");
  m_maxEvents = config   ("events", 32U);
  m_filter    = config   ("filter", false);
}

//--------------
// Destructor --
//--------------
CSPadTest::~CSPadTest () {}


/// Method which is called once at the beginning of the job
//void CSPadTest::beginJob(Event& evt, Env& env) {}


/// Method which is called at the beginning of the run
void CSPadTest::beginRun(Event& evt, Env& env) 
{

  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_src);

  if (config2.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV2:";
      str << "\n  concentratorVersion = " << config2->concentratorVersion();
      str << "\n  runDelay            = " << config2->runDelay();
      str << "\n  eventCode           = " << config2->eventCode();
      str << "\n  inactiveRunMode     = " << config2->inactiveRunMode();
      str << "\n  activeRunMode       = " << config2->activeRunMode();
      str << "\n  tdi                 = " << config2->tdi();
      str << "\n  payloadSize         = " << config2->payloadSize();
      str << "\n  badAsicMask0        = " << config2->badAsicMask0();
      str << "\n  badAsicMask1        = " << config2->badAsicMask1();
      str << "\n  asicMask            = " << config2->asicMask();
      str << "\n  quadMask            = " << config2->quadMask();
      str << "\n  numAsicsRead        = " << config2->numAsicsRead();
      str << "\n  numSect             = " << config2->numSect();
      str << "\n  numQuads            = " << config2->numQuads();

      std::vector<int> v_quads_shape = config2->quads_shape();
      str << "\n  v_quads_shape.size()= " << v_quads_shape.size();
      str << "\n  v_quads_shape[0]    = " << v_quads_shape[0];

      for (uint32_t q = 0; q < Psana::CsPad::MaxQuadsPerSensor; ++ q) {
        str << "\n  roiMask       (" << q << ") = " << config2->roiMask(q);
        str << "\n  numAsicsStored(" << q << ") = " << config2->numAsicsStored(q);
      }

      str << "\n  MaxQuadsPerSensor = " << Psana::CsPad::MaxQuadsPerSensor;
      str << "\n  ASICsPerQuad      = " << Psana::CsPad::ASICsPerQuad;
      str << "\n  RowsPerBank       = " << Psana::CsPad::RowsPerBank;
      str << "\n  FullBanksPerASIC  = " << Psana::CsPad::FullBanksPerASIC;
      str << "\n  BanksPerASIC      = " << Psana::CsPad::BanksPerASIC;
      str << "\n  ColumnsPerASIC    = " << Psana::CsPad::ColumnsPerASIC;
      str << "\n  MaxRowsPerASIC    = " << Psana::CsPad::MaxRowsPerASIC;
      str << "\n  PotsPerQuad       = " << Psana::CsPad::PotsPerQuad;
      str << "\n  TwoByTwosPerQuad  = " << Psana::CsPad::TwoByTwosPerQuad;
      str << "\n  SectorsPerQuad    = " << Psana::CsPad::SectorsPerQuad;

    }
  } 

}


/// Method which is called at the beginning of the calibration cycle
//void CSPadTest::beginCalibCycle(Event& evt, Env& env) {}


/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void CSPadTest::event(Event& evt, Env& env)
{
  // get event ID

  /*
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (not eventId.get()) {
    MsgLog(name(), info, "event ID not found");
  } else {
    MsgLog(name(), info, "event ID: " << *eventId);
  }
  */


  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {
    WithMsgLog(name(), info, str) {
      str << "CsPad::DataV2:";
      int nQuads = data2->quads_shape()[0];
        str << "\n    nQuads = " << nQuads ;
      for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);
        str << "\n  Element #" << q ;
        str << "\n    virtual_channel = " << el.virtual_channel() ;
        str << "\n    lane = "            << el.lane() ;
        str << "\n    tid = "             << el.tid() ;
        str << "\n    acq_count = "       << el.acq_count() ;
        str << "\n    op_code = "         << el.op_code() ;
        str << "\n    quad = "            << el.quad() ;
        str << "\n    seq_count = "       << el.seq_count() ;
        str << "\n    ticks = "           << el.ticks() ;
        str << "\n    fiducials = "       << el.fiducials() ;
        str << "\n    frame_type = "      << el.frame_type() ;
        str << "\n    sb_temp = [ ";

        const uint16_t* sb_temp = el.sb_temp();
        std::copy(sb_temp, sb_temp+10, // +Psana::CsPad::ElementV2::Nsbtemp, 
            std::ostream_iterator<uint16_t>(str, " "));
        str << "]";

        std::vector<int> v_image_shape = el.data_shape();
        str << "\n    v_image_shape = "   << v_image_shape[0]
                                  << "  " << v_image_shape[1]
                                  << "  " << v_image_shape[2];

	m_Nquads = v_image_shape[0];
        m_Nrows  = v_image_shape[1];
        m_Ncols  = v_image_shape[2];
      }
    } //WithMsgLog(


    int nQuads = data2->quads_shape()[0];
    cout << "\n    nQuads = " << nQuads << endl;
    for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);
        cout  << "\n  Quad / Element #" << q << endl;

	const uint16_t* data = el.data();


        //uint16_t data2d[185][388];
        //uint16_t* data2d = new uint16_t[m_Nrows*m_Ncols];
	//memcpy (data2d, data, m_Nrows*m_Ncols*sizeof(uint16_t));
	//iterateOverData(data2d);
	//delete [] data2d;

        Image2D<uint16_t>* pair_arr_image = new Image2D<uint16_t>(data,m_Nrows,m_Ncols);
        pair_arr_image -> printImage();

        //uint16_t data2d[185][388];

    }


  } // if (data2.get())



  // this is how to skip event (all downstream modules will not be called)
  if (m_filter && m_count % 10 == 0) skip();
 

  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) stop();
  // increment event counter
  m_count++;
  cout << "m_count=" << m_count << endl;

}


void CSPadTest::iterateOverData(const uint16_t data[][388]) // [185][388] for one quad
//void CSPadTest::iterateOverData(const uint16_t* data)
{
  cout << "\n  sizeof(data)       = "  << sizeof(data) << endl;
  //str << "\n  sizeof(data[0])    = "  << sizeof(data[0]);
  //str << "\n  sizeof(data[0][0]) = "  << sizeof(data[0][0]);

	for (int row = 0; row < m_Nrows; row+=20) {
	  for (int col = 0; col < m_Ncols; col+=20) {

	    //cout << data[row*m_Ncols + col] << "  ";
	    cout << data[row][col] << "  ";

	  }
	    cout << endl;
	}
}




  
/// Method which is called at the end of the calibration cycle
//void CSPadTest::endCalibCycle(Event& evt, Env& env) {}


/// Method which is called at the end of the run
//void CSPadTest::endRun(Event& evt, Env& env) {}


/// Method which is called once at the end of the job
void CSPadTest::endJob(Event& evt, Env& env) {

  uint16_t arr2d[3][4] = { { 0, 1, 2, 3 },
                           { 4, 5, 6, 7 },
                           { 8, 9, 10,11} 
                         };
  
		     int nrows=3;
		     int ncols=4;

  Image2D<uint16_t>* test_image = new Image2D<uint16_t>(&arr2d[0][0],nrows,ncols);  
                     test_image -> printEntireImage();
                     test_image -> printEntireImage(1);
                     test_image -> printEntireImage(2);
                     test_image -> printEntireImage(3);


		     int gap_ncols = 2;

  ImageCSPad2x1<uint16_t>* image_2x1 = new ImageCSPad2x1<uint16_t>(&arr2d[0][0], gap_ncols, nrows, ncols);  
	                   image_2x1 -> printEntireImage(0);
	                   image_2x1 -> printEntireImage(1);
	                   image_2x1 -> printEntireImage(2);
	                   image_2x1 -> printEntireImage(3);

}


} // namespace CSPadImage
