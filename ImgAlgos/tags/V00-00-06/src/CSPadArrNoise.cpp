//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrNoise...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrNoise.h"

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

PSANA_MODULE_FACTORY(CSPadArrNoise)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

CSPadArrNoise::CSPadArrNoise (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_fracFile()
  , m_maskFile()
  , m_rmin()
  , m_dr()
  , m_SoNThr()
  , m_frac_noisy_imgs()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src         = configStr("source",     "DetInfo(:Cspad)");
  m_key             = configStr("key",        "");                 //"calibrated"
  m_fracFile      = configStr("fracfile", "cspad-pix-frac.dat");
  m_maskFile        = configStr("maskfile", "cspad-pix-mask.dat");
  m_rmin            = config   ("rmin",              3);
  m_dr              = config   ("dr",                2);
  m_SoNThr          = config   ("SoNThr",            3);
  m_frac_noisy_imgs = config   ("frac_noisy_imgs", 0.1); 
  m_print_bits      = config   ("print_bits",        0);

  // initialize arrays
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);

  resetStatArrays();
}

//--------------
// Destructor --
//--------------
CSPadArrNoise::~CSPadArrNoise ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadArrNoise::beginJob(Event& evt, Env& env)
{
  evaluateVectorOfIndexesForMedian();

  if( m_print_bits &   1 ) printInputParameters();
  if( m_print_bits &  64 ) printVectorOfIndexesForMedian();
  if( m_print_bits &  64 ) printMatrixOfIndexesForMedian();
}

/// Method which is called at the beginning of the run
void 
CSPadArrNoise::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  //std::string src = configStr("source", "DetInfo(:Cspad)");
  int count = 0;
  
  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(m_str_src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff; }
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_str_src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config2->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(m_str_src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config3->roiMask(i); }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CSPad configuration objects found. Terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CSPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CSPad object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found CSPad configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed CSPad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found CSPad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}



/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrNoise::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrNoise::event(Event& evt, Env& env)
{
  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_str_src, m_key, &m_src);
  if (data1.get()) {

    ++ m_count;
    //setCollectionMode();
    
    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStatInQuad(quad.quad(), data.data());
    }    
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_str_src, m_key, &m_src);
  if (data2.get()) {

    ++ m_count;
    //setCollectionMode();
    
    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStatInQuad(quad.quad(), data.data());
    } 
  }
  //if( m_print_bits & 16 ) printEventId(evt);
  if( m_print_bits & 32 ) printTimeStamp(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadArrNoise::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrNoise::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrNoise::endJob(Event& evt, Env& env)
{
  procStatArrays();
  saveCSPadArrayInFile<float>( m_fracFile, m_status );
  saveCSPadArrayInFile<uint16_t>( m_maskFile,   m_mask );  //or &m_mask[0][0][0][0] );
}

//--------------------

/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
CSPadArrNoise::procStatArrays()
{
  if( m_print_bits & 4 ) MsgLog(name(), info, "Process statistics for collected total " << m_count << " events");

  unsigned long  npix_noisy = 0;
  unsigned long  npix_total = 0;
  
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            npix_total ++;
	    unsigned stat = m_stat[iq][is][ic][ir];
	    
	    if(m_count > 0) { 
	      
	      float fraction_of_noisy = float(stat) / m_count; 

              m_status[iq][is][ic][ir] = fraction_of_noisy;

	      if (fraction_of_noisy < m_frac_noisy_imgs) {

                m_mask[iq][is][ic][ir] = 1; 
	      }
	      else
	      {
                npix_noisy ++;
	      }
	      
              //if (stat > 0) cout << "q,s,c,r=" << iq << " " << is << " " << ic << " " << ir
	      //                 << " stat, total=" << stat << " " << m_count << endl;

            } 
          }
        }
      }
    }

    cout << "Nnoisy, Ntotal, Nnoisy/Ntotal pixels =" << npix_noisy << " " << npix_total  << " " << double(npix_noisy)/npix_total << endl;
}

//--------------------

/// Save 4-d array of CSPad structure in file
template <typename T>
void 
CSPadArrNoise::saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows])
{  
  if (not fname.empty()) {
    if( m_print_bits & 8 ) MsgLog(name(), info, "Save CSPad-shaped array in file " << fname.c_str());
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
CSPadArrNoise::resetStatArrays()
{
  std::fill_n(&m_stat  [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0 );
  std::fill_n(&m_mask  [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0 );
  std::fill_n(&m_status[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
}

//--------------------
/// Check the event counter and deside what to do next accumulate/change mode/etc.
/*
void 
CSPadArrNoise::setCollectionMode()
{
  if (m_count == 1 ) {
    m_gate_width = 0;
    resetStatArrays();
    if( m_print_bits & 2 ) MsgLog(name(), info, "Stage 0: Event = " << m_count << " Begin to collect statistics without gate.");
  }
...
}
*/

//--------------------
/// Collect statistics
/// Loop over all 2x1 sections available in the event 
void 
CSPadArrNoise::collectStatInQuad(unsigned quad, const int16_t* data)
{
  //cout << "collectStat for quad =" << quad << endl;

  int ind_in_arr = 0;
  for (unsigned sect = 0; sect < MaxSectors; ++ sect) {
    if (m_segMask[quad] & (1 << sect)) {
     
      const int16_t* sectData = data + ind_in_arr*SectorSize;

      collectStatInSect(quad, sect, sectData);
      
      ++ind_in_arr;
    }
  }
}

//--------------------
/// Collect statistics in one section
/// Loop over one 2x1 section pixels, evaluate S/N and count statistics above threshold 
void 
CSPadArrNoise::collectStatInSect(unsigned quad, unsigned sect, const int16_t* sectData)
{
  for (int ic = 0; ic != NumColumns; ++ ic) {
    for (int ir = 0; ir != NumRows; ++ ir) {

      MedianResult res = evaluateSoNForPixel(ic,ir,sectData);

      if ( abs( res.SoN ) > m_SoNThr ) m_stat[quad][sect][ic][ir] ++;

      /*
      m_bkgd_arr  [quad][sect][ic][ir] = res.avg;
      m_rms_arr   [quad][sect][ic][ir] = res.rms;
      m_signal_arr[quad][sect][ic][ir] = res.signal;
      m_SoN_arr   [quad][sect][ic][ir] = res.SoN;
      if ( res.SoN > m_SoNThr ) m_[quad][sect][ic][ir] = 1;
      */
    }
  }
}

//--------------------
/// Evaluate vector of indexes for mediane algorithm
/// The area of pixels for the mediane algorithm is defined as a ring from m_rmin to m_rmin + m_dr
void 
CSPadArrNoise::evaluateVectorOfIndexesForMedian()
{
  v_indForMediane.clear();

  TwoIndexes inds;
  int indmax = int(m_rmin + m_dr);
  int indmin = -indmax;

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      float r = std::sqrt( float(i*i + j*j) );
      if ( r < m_rmin || r > m_rmin + m_dr ) continue;
      inds.i = i;
      inds.j = j;
      v_indForMediane.push_back(inds);
    }
  }
}

//--------------------

void 
CSPadArrNoise::printMatrixOfIndexesForMedian()
{
  int indmax = int(m_rmin + m_dr);
  int indmin = -indmax;

  cout << "CSPadArrNoise::printMatrixOfIndexesForMedian():" << endl;
  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      float r = std::sqrt( float(i*i + j*j) );
      int status = ( r < m_rmin || r > m_rmin + m_dr ) ? 0 : 1;
      if (i==0 && j==0) cout << " +";
      else              cout << " " << status;
    }
    cout << endl;
  }
}

//--------------------
/// Print vector of indexes for mediane algorithm
void 
CSPadArrNoise::printVectorOfIndexesForMedian()
{
  std::cout << "CSPadArrNoise::printVectorOfIndexesForMedian():" << std::endl;
  int n_pairs_in_line=0;
  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {

    cout << " (" << ij->i << "," << ij->j << ")";
    if ( ++n_pairs_in_line > 9 ) {cout << "\n"; n_pairs_in_line=0;}
  }   
  cout << "\nVector size: " << v_indForMediane.size() << endl;
}

//--------------------
/// Apply median algorithm for one pixel
MedianResult
CSPadArrNoise::evaluateSoNForPixel(unsigned col, unsigned row, const int16_t* sectData)
{

  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;

  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {
    int ic = col + (ij->i);
    int ir = row + (ij->j);

    if(ic < 0)           continue;
    if(ic > NumColumns1) continue;
    if(ir < 0)           continue;
    if(ir > NumRows1)    continue;

    double  amp = sectData[ir + ic*NumRows];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }

  MedianResult res = {0,0,0,0};

  if ( sum0 > 0 ) {
    res.avg = sum1/sum0;                                // Averaged background level
    res.rms = std::sqrt( sum2/sum0 - res.avg*res.avg ); // RMS os the background around peak
    res.signal = sectData[row + col*NumRows] - res.avg; // Signal above the background
    if (res.rms>0) res.SoN = res.signal/res.rms;        // S/N ratio
  }

  return res;
}

//--------------------

// Print input parameters
void 
CSPadArrNoise::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source            : " << m_str_src
        << "\n key               : " << m_key      
        << "\n m_fracFile        : " << m_fracFile    
        << "\n m_maskFile        : " << m_maskFile    
        << "\n m_rmin            : " << m_rmin    
        << "\n m_dr              : " << m_dr     
        << "\n m_SoNThr          : " << m_SoNThr     
        << "\n m_frac_noisy_imgs : " << m_frac_noisy_imgs    
        << "\n print_bits : "        << m_print_bits
        << "\n";     

    log << "\n MaxQuads   : " << MaxQuads    
        << "\n MaxSectors : " << MaxSectors  
        << "\n NumColumns : " << NumColumns  
        << "\n NumRows    : " << NumRows     
        << "\n SectorSize : " << SectorSize  
        << "\n";
  }
}

//--------------------

void 
CSPadArrNoise::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
  }
}

//--------------------

void 
CSPadArrNoise::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, " Run="   <<  eventId->run()
                       << " Event=" <<  m_count 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------

} // namespace ImgAlgos
