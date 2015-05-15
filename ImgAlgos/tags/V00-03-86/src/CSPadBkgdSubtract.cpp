//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadBkgdSubtract...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadBkgdSubtract.h"

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
PSANA_MODULE_FACTORY(CSPadBkgdSubtract)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadBkgdSubtract::CSPadBkgdSubtract (const std::string& name)
  : CSPadBaseModule(name, "inputKey")
  , m_outkey()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_outkey      = configStr("outputKey", "bkgd_subtracted");
  m_fname       = configStr("bkgd_fname", "cspad_background.dat");
  m_norm_sector = config("norm_sector", 0); // from 0 to 7
  m_print_bits  = config("print_bits", 0);

  std::fill_n(&m_common_mode[0], int(MaxSectors), float(0));
}

//--------------
// Destructor --
//--------------
CSPadBkgdSubtract::~CSPadBkgdSubtract ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadBkgdSubtract::beginJob(Event& evt, Env& env)
{
  getBkgdArray();
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 8 ) printBkgdArray();
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadBkgdSubtract::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadBkgdSubtract::event(Event& evt, Env& env)
{
  normalizeBkgd(evt);
  subtractBkgd(evt);
}

/// Method which is called at the end of the calibration cycle
void 
CSPadBkgdSubtract::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadBkgdSubtract::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadBkgdSubtract::endJob(Event& evt, Env& env)
{
}

//--------------------

// Print input parameters
void 
CSPadBkgdSubtract::printInputParameters()
{
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << sourceConfigured()
        << "\n inkey      : " << inputKey()
        << "\n outkey     : " << m_outkey      
        << "\n fname      : " << m_fname    
        << "\n norm_sector: " << m_norm_sector 
        << "\n print_bits : " << m_print_bits
        << "\n";
  }
}

//--------------------

void 
CSPadBkgdSubtract::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    //MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
    MsgLog( name(), info, "Event="  << m_count << " time: " << stringTimeStamp(evt) );
  }
}

//--------------------

std::string
CSPadBkgdSubtract::stringTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return (eventId->time()).asStringFormat("%Y%m%dT%H:%M:%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  }
  return std::string("Time-stamp-is-unavailable");
}

//--------------------

void
CSPadBkgdSubtract::getBkgdArray()
{
  m_bkgd = new ImgAlgos::CSPadBackgroundV1(m_fname);
}

//--------------------

void 
CSPadBkgdSubtract::printBkgdArray()
{
  m_bkgd -> print(); 
}

//--------------------

/// Normalize the background
void 
CSPadBkgdSubtract::normalizeBkgd(Event& evt)
{
  m_norm_factor = 0;

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(source(), inputKey());
  if (data1.get()) {

    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq); // get quad object 
      const ndarray<const int16_t,3>& data = quad.data();    // get data for quad

      double quad_norm_factor = normQuadBkgd(quad.quad(), data.data());
      if( quad_norm_factor > 0 ) m_norm_factor += quad_norm_factor; 
    }    
    m_norm_factor = ( nQuads ) ? m_norm_factor /= nQuads : 0; 
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(source(), inputKey());
  if (data2.get()) {

    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq); // get quad object 
      const ndarray<const int16_t,3>& data = quad.data();    // get data for quad

      double quad_norm_factor = normQuadBkgd(quad.quad(), data.data());
      if( quad_norm_factor > 0 ) m_norm_factor += quad_norm_factor; 
    } 
    m_norm_factor = ( nQuads > 0 ) ? m_norm_factor /= nQuads : 0; 
  }

}

//--------------------

/// Subtract the background and save array in the event
void 
CSPadBkgdSubtract::subtractBkgd(Event& evt)
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
  if( m_print_bits & 4 ) MsgLog( name(), info, "Event="  << m_count << " Norm. factor="  << m_norm_factor ); 
}

//--------------------

/// Process data for all sectors in quad
void 
CSPadBkgdSubtract::processQuad(unsigned quad, const int16_t* data, int16_t* corrdata)
{
  //cout << "processQuad =" << quad << endl;

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
      // beginning of the segment data
      const int16_t* sectData = data     + ind_in_arr*SectorSize;
      int16_t*       corrData = corrdata + ind_in_arr*SectorSize;
      double*        sectBkgd = m_bkgd->getBackground(quad,sect);

      // Background subtraction
      for (int i = 0; i < SectorSize; ++ i) {

        corrData[i] = sectData[i] - int16_t(m_norm_factor * sectBkgd[i]);
      }                
      ++ind_in_arr;
    }
  }  
}

//--------------------

/// Process data for all sectors in quad
double 
CSPadBkgdSubtract::normQuadBkgd(unsigned quad, const int16_t* data)
{
  //cout << "normQuadBkgd: quad =" << quad;

  double sum_data   = 0;
  double sum_bkgd   = 0;  

  int ind_in_arr = 0;
  for (unsigned sect = 0; sect < MaxSectors; ++ sect) {
    if ( segMask(quad) & (1<<sect) ) {
      if ( sect == m_norm_sector ) {
     
        // beginning of the segment data
        const int16_t* sectData = data     + ind_in_arr*SectorSize;
        double*        sectBkgd = m_bkgd->getBackground(quad,sect);

        // Background subtraction
        for (int i = 0; i < SectorSize; ++ i) {
          sum_data += sectData[i];
          sum_bkgd += sectBkgd[i];
        }                

	break;
      }
      ++ind_in_arr;
    }
  }  

  double quad_norm_factor = (sum_bkgd>0) ? sum_data/sum_bkgd : 0;
  //cout << "  sum_data =" << sum_data <<  "  sum_bkgd =" << sum_bkgd <<  "  f =" << quad_norm_factor << endl; 
  return quad_norm_factor;
}

//--------------------

} // namespace ImgAlgos
