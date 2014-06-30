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

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

CSPadArrNoise::CSPadArrNoise (const std::string& name)
  : CSPadBaseModule(name)
  , m_fracFile()
  , m_maskFile()
  , m_rmin()
  , m_dr()
  , m_SoNThr()
  , m_frac_noisy_imgs()
  , m_print_bits()
{
  // get the values from configuration or use defaults
  m_fracFile        = configStr("fracfile", "cspad-pix-frac.dat");
  m_maskFile        = configStr("maskfile", "cspad-pix-mask.dat");
  m_rmin            = config   ("rmin",              3);
  m_dr              = config   ("dr",                2);
  m_SoNThr          = config   ("SoNThr",            3);
  m_frac_noisy_imgs = config   ("frac_noisy_imgs", 0.1); 
  m_print_bits      = config   ("print_bits",        0);

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
  //if( m_print_bits & 16 ) printEventId(evt);
   if( m_print_bits & 32 ) printTimeStamp(evt);

   if ( procEventForType<Psana::CsPad::DataV1, CsPad::ElementV1> (evt) ) return;
   if ( procEventForType<Psana::CsPad::DataV2, CsPad::ElementV2> (evt) ) return;

   MsgLog(name(), warning, "event(...): Psana::CsPad::DataV# / ElementV# is not available in this event.");
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
  saveCSPadArrayInFile<uint16_t>( m_maskFile, m_mask );  //or &m_mask[0][0][0][0] );
}

//--------------------
/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
CSPadArrNoise::procStatArrays()
{
  if( m_print_bits & 4 ) MsgLog(name(), info, "Process statistics for collected total " << counter() << " events");

  unsigned long  npix_noisy = 0;
  unsigned long  npix_total = 0;
  
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            npix_total ++;
	    unsigned stat = m_stat[iq][is][ic][ir];
	    
	    if(counter() > 0) { 
	      
	      float fraction_of_noisy = float(stat) / counter(); 

              m_status[iq][is][ic][ir] = fraction_of_noisy;

	      if (fraction_of_noisy < m_frac_noisy_imgs) {

                m_mask[iq][is][ic][ir] = 1; 
	      }
	      else
	      {
                npix_noisy ++;
	      }
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
/// Implementation for abstract method from CSPadBaseModule.h
/// Collect statistics
/// Loop over all 2x1 sections available in the event 
void 
CSPadArrNoise::procQuad(unsigned quad, const int16_t* data)
{
  //cout << "procQuad: collect statistics for quad =" << quad << endl;

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
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
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source            : " << sourceConfigured()
        << "\n key               : " << inputKey()
        << "\n m_fracFile        : " << m_fracFile    
        << "\n m_maskFile        : " << m_maskFile    
        << "\n m_rmin            : " << m_rmin    
        << "\n m_dr              : " << m_dr     
        << "\n m_SoNThr          : " << m_SoNThr     
        << "\n m_frac_noisy_imgs : " << m_frac_noisy_imgs    
        << "\n print_bits : "        << m_print_bits
        << "\n";     
  }
}

//--------------------
void 
CSPadArrNoise::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << counter() << " ID: " << *eventId);
  }
}

//--------------------

void 
CSPadArrNoise::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, " Run="   <<  eventId->run()
                       << " Event=" <<  counter() 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------

} // namespace ImgAlgos
