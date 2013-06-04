//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrAverage...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrAverage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(CSPadArrAverage)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadArrAverage::CSPadArrAverage (const std::string& name)
  : CSPadBaseModule(name)
  , m_aveFile()
  , m_rmsFile()
  , m_print_bits()
  , m_nev_stage1()
  , m_nev_stage2()
  , m_gate_width1()
  , m_gate_width2()
{
  // get the values from configuration or use defaults
  m_aveFile = configStr("avefile", "cspad-ave.dat");
  m_rmsFile = configStr("rmsfile", "cspad-rms.dat");
  m_nev_stage1  = config("evts_stage1", 1<<31U);
  m_nev_stage2  = config("evts_stage2",    100);
  m_gate_width1 = config("gate_width1",      0); 
  m_gate_width2 = config("gate_width2",      0); 
  m_print_bits  = config("print_bits",       0);

  m_gate_width = 0;
}

//--------------
// Destructor --
//--------------
CSPadArrAverage::~CSPadArrAverage ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadArrAverage::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1<<0 ) printInputParameters();
}


/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrAverage::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrAverage::event(Event& evt, Env& env)
{
   if( m_print_bits & 1<<4 ) printEventId(evt);

   if ( procEventForType<Psana::CsPad::DataV1, CsPad::ElementV1> (evt) ) return;
   if ( procEventForType<Psana::CsPad::DataV2, CsPad::ElementV2> (evt) ) return;

   MsgLog(name(), warning, "event(...): Psana::CsPad::DataV# / ElementV# is not available in this event.");
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadArrAverage::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrAverage::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrAverage::endJob(Event& evt, Env& env)
{
  procStatArrays();
  saveCSPadArrayInFile( m_aveFile, m_ave ); // &m_ave[0][0][0][0] );
  saveCSPadArrayInFile( m_rmsFile, m_rms ); // &m_rms[0][0][0][0] );
}

//--------------------
/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
CSPadArrAverage::procStatArrays()
{
  if( m_print_bits & 1<<2 ) MsgLog(name(), info, "Process statistics for collected total " << counter() << " events");
  
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

	    double stat  = m_stat[iq][is][ic][ir];
            if (stat > 1) {
              double ave   = m_sum[iq][is][ic][ir] / stat;
	      m_ave[iq][is][ic][ir] = ave;
              m_rms[iq][is][ic][ir] = std::sqrt(m_sum2[iq][is][ic][ir] / stat - ave*ave);
            } 
            else 
            {
	      m_ave[iq][is][ic][ir] = 0;
	      m_rms[iq][is][ic][ir] = 0;
            }
          }
        }
      }
    }
}

//--------------------
/// Save 4-d array of CSPad structure in file
void 
CSPadArrAverage::saveCSPadArrayInFile(std::string& fname, double arr[MaxQuads][MaxSectors][NumColumns][NumRows])
{  
  if (not fname.empty()) {
    if( m_print_bits & 1<<3 ) MsgLog(name(), info, "Save CSPad-shaped array in file " << fname.c_str());
    std::ofstream out(fname.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            out << arr[iq][is][ic][ir] << ' ';
          }
          out << '\n';
        }
      }
    }
    out.close();
  }
}

//--------------------
/// Reset arrays for statistics accumulation
void
CSPadArrAverage::resetStatArrays()
{
  std::fill_n(&m_stat[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0 );
  std::fill_n(&m_sum [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
  std::fill_n(&m_sum2[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
}

//--------------------
/// Implementation for abstract method from CSPadBaseModule.h
/// Check the event counter and deside what to do next accumulate/change mode/etc.
void 
CSPadArrAverage::initData()
{
  // Set the statistics collection mode without gate
  if (counter() == 1 ) {
    m_gate_width = 0;
    resetStatArrays();
    if( m_print_bits & 1<<1 ) MsgLog(name(), info, "Stage 0: Event = " << counter() << " Begin to collect statistics without gate.");
  }

  // Change the statistics collection mode for gated stage 1
  else if (counter() == m_nev_stage1 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width1;
    if( m_print_bits & 1<<1 ) MsgLog(name(), info, "Stage 1: Event = " << counter() << " Begin to collect statistics with gate =" << m_gate_width);
  } 

  // Change the statistics collection mode for gated stage 2
  else if (counter() == m_nev_stage1 + m_nev_stage2 ) {
    procStatArrays();
    resetStatArrays();
    m_gate_width = m_gate_width2;
    if( m_print_bits & 1<<1 ) MsgLog(name(), info, "Stage 2: Event = " << counter() << " Begin to collect statistics with gate =" << m_gate_width);
  }
}

//--------------------
/// Implementation for abstract method from CSPadBaseModule.h
/// Collect statistics
void 
CSPadArrAverage::procQuad(unsigned quad, const int16_t* data)
{
  //cout << "procQuad for quad =" << quad << endl;

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
      // beginning of the segment data
      unsigned* stat = &m_stat[quad][sect][0][0];
      double*   sum  = &m_sum [quad][sect][0][0];
      double*   sum2 = &m_sum2[quad][sect][0][0];
      double*   ave  = &m_ave [quad][sect][0][0];
      //double*   rms  = &m_rms [quad][sect][0][0];
      const int16_t* segData = data + ind_in_arr*SectorSize;

      // sum
      for (int i = 0; i < SectorSize; ++ i) {

	double amp = double(segData[i]);
	if ( m_gate_width > 0 && std::abs(amp-ave[i]) > m_gate_width ) continue; // gate_width -> n_rms_gate_width * rms[i]

        stat[i] ++;
        sum [i] += amp;
        sum2[i] += amp*amp;
      }          
      
      ++ind_in_arr;
    }
  }
}

//--------------------
// Print input parameters
void 
CSPadArrAverage::printInputParameters()
{
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << sourceConfigured()
        << "\n key        : " << inputKey()
        << "\n m_aveFile  : " << m_aveFile    
        << "\n m_rmsFile  : " << m_rmsFile    
        << "\n print_bits : " << m_print_bits
        << "\n evts_stage1: " << m_nev_stage1   
        << "\n evts_stage2: " << m_nev_stage2  
        << "\n gate_width1: " << m_gate_width1 
        << "\n gate_width2: " << m_gate_width2 
        << "\n";     
  }
}

//--------------------
void 
CSPadArrAverage::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << counter() << " ID: " << *eventId);
  }
}

//--------------------

} // namespace ImgAlgos
