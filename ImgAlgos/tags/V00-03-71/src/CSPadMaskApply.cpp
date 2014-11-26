//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadMaskApply...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadMaskApply.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "PSEvt/EventId.h"
#include "cspad_mod/DataT.h"
#include "cspad_mod/ElementT.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(CSPadMaskApply)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadMaskApply::CSPadMaskApply (const std::string& name)
  : CSPadBaseModule(name, "inkey", "")
  , m_outkey()
  , m_fname()
  , m_masked_amp()
  , m_mask_control_bits()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_outkey             = configStr("outkey",     "mask_applyed");
  m_fname              = configStr("mask_fname", "cspad_mask.dat");
  m_masked_amp         = config   ("masked_amp", 0);
  m_mask_control_bits  = config   ("mask_control_bits", 1);
  m_print_bits         = config   ("print_bits", 0);

  std::fill_n(&m_common_mode[0], int(MaxSectors), float(0));
}

//--------------
// Destructor --
//--------------
CSPadMaskApply::~CSPadMaskApply ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadMaskApply::beginJob(Event& evt, Env& env)
{
  getMaskArray();
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 4 ) printMaskStatistics();
  if( m_print_bits & 8 ) printMaskArray();
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadMaskApply::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadMaskApply::event(Event& evt, Env& env)
{
  applyMask(evt);
}

/// Method which is called at the end of the calibration cycle
void 
CSPadMaskApply::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadMaskApply::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadMaskApply::endJob(Event& evt, Env& env)
{
}

//--------------------

// Print input parameters
void 
CSPadMaskApply::printInputParameters()
{
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source            : " << sourceConfigured()
        << "\n inkey             : " << inputKey()
        << "\n outkey            : " << m_outkey      
        << "\n fname             : " << m_fname    
        << "\n masked_amp        : " << m_masked_amp
        << "\n mask_control_bits : " << m_mask_control_bits
        << "\n print_bits        : " << m_print_bits
        << "\n";     
  }
}

//--------------------

void 
CSPadMaskApply::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    //MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
    MsgLog( name(), info, "Event="  << m_count << " time: " << stringTimeStamp(evt) );
  }
}

//--------------------

std::string
CSPadMaskApply::stringTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return (eventId->time()).asStringFormat("%Y%m%dT%H:%M:%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  }
  return std::string("Time-stamp-is-unavailable");
}

//--------------------

void
CSPadMaskApply::getMaskArray()
{
  m_mask = new ImgAlgos::CSPadMaskV1(m_fname);
}

//--------------------

void 
CSPadMaskApply::printMaskArray()
{
  m_mask -> print(); 
}

//--------------------

void 
CSPadMaskApply::printMaskStatistics()
{
  m_mask -> printMaskStatistics(); 
}

//--------------------

/// Apply the mask and save array in the event
void 
CSPadMaskApply::applyMask(Event& evt)
{
  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(source(), inputKey());
  if (data1.get()) {

    ++ m_count;

    shared_ptr<cspad_mod::DataV1> newobj(new cspad_mod::DataV1());

    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq); // get quad object 
      const ndarray<const int16_t,3>& data = quad.data();    // get data for quad

      int16_t* corrdata = new int16_t[data.size()];  // allocate memory for corrected quad-array 
      //int16_t* corrdata = &m_corrdata[iq][0][0][0];        // allocate memory for corrected quad-array 
      processQuad(quad.quad(), data.data(), corrdata); // process event for quad

      newobj->append(new cspad_mod::ElementV1(quad, corrdata, m_common_mode));
    }    
    evt.put<Psana::CsPad::DataV1>(newobj, source(), m_outkey); // put newobj in event
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(source(), inputKey());
  if (data2.get()) {

    ++ m_count;

    shared_ptr<cspad_mod::DataV2> newobj(new cspad_mod::DataV2());
    
    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq); // get quad object 
      const ndarray<const int16_t,3>& data = quad.data();    // get data for quad

      int16_t* corrdata = new int16_t[data.size()];  // allocate memory for corrected quad-array 
      //int16_t* corrdata = &m_corrdata[iq][0][0][0];        // allocate memory for corrected quad-array 
      processQuad(quad.quad(), data.data(), corrdata); // process event for quad

      newobj->append(new cspad_mod::ElementV2(quad, corrdata, m_common_mode)); 
    } 
    evt.put<Psana::CsPad::DataV2>(newobj, source(), m_outkey); // put newobj in event
  }

  if( m_print_bits & 2 ) printEventId(evt);
}

//--------------------

/// Process data for all sectors in quad
void 
CSPadMaskApply::processQuad(unsigned quad, const int16_t* data, int16_t* corrdata)
{
  //cout << "processQuad =" << quad << endl;

  //if( m_mask_control_bits == 0 ) { corrdata = data; return; }

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
      // beginning of the segment data
      const int16_t* sectData = data     + ind_in_arr*SectorSize;
      int16_t*       corrData = corrdata + ind_in_arr*SectorSize;
      uint16_t*      sectMask = m_mask->getMask(quad,sect);

      if( m_mask_control_bits & 1 ) { 
        // Apply mask from file
        for (int i = 0; i < SectorSize; ++ i) {
          corrData[i] = (sectMask[i] != 0) ? sectData[i] : m_masked_amp;
        }                
      } else {      
        // DO NOT apply mask from file
        for (int i = 0; i < SectorSize; ++ i) {
          corrData[i] = sectData[i];
        }                
      }


      if( m_mask_control_bits & 2 ) { 
        // Mask 2 long edges of 2x1
        int c0 = 0; 
        int cN = NumColumns-1; 
        for (int r = 0; r < NumRows; r++ ) { // NumRows = 388
          corrData[c0*NumRows + r] = m_masked_amp;
          corrData[cN*NumRows + r] = m_masked_amp;
        }
      }

        
      if( m_mask_control_bits & 4 ) { 
        // Mask 2 short edges of 2x1
        int r0 = 0; 
        int rN = NumRows-1; 
        for (int c = 0; c < NumColumns; c++ ) { // NumColumns = 185
  	  corrData[c*NumRows + r0] = m_masked_amp;
  	  corrData[c*NumRows + rN] = m_masked_amp;
        }
      }

        
      if( m_mask_control_bits & 8 ) { 
        // Mask 2 raws in the middle of 2x1 with wide pixels
        int rHL = 388/2-1; 
        int rHU = 388/2; 
        for (int c = 0; c < NumColumns; c++ ) { // NumColumns = 185
          corrData[c*NumRows + rHL] = m_masked_amp;
          corrData[c*NumRows + rHU] = m_masked_amp;
        }
      }

      ++ind_in_arr;
    }
  }  
}

//--------------------

} // namespace ImgAlgos
