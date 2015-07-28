//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AlgArrProc
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AlgArrProc.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <cmath>     // floor, ceil
//#include <iomanip>   // for std::setw
#include <sstream>   // for stringstream

namespace ImgAlgos {

  typedef AlgArrProc::wind_t wind_t;
  //typedef AlgArrProc::Peak   Peak;

//--------------------
// Constructors --
//--------------------

//AlgArrProc::AlgArrProc(ndarray<const wind_t,2>* p_nda_winds, const unsigned& pbits)
//AlgArrProc::AlgArrProc(ndarray<const wind_t,2>& nda_winds, const unsigned& pbits)

AlgArrProc::AlgArrProc()
  : m_pbits(0)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  v_winds.clear();
  setPeakSelectionPars(); // default init

  if(m_pbits & 1) printInputPars();
}

//--------------------

AlgArrProc::AlgArrProc(const unsigned& pbits)
  : m_pbits(pbits)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  v_winds.clear();
  setPeakSelectionPars(); // default init
}

//--------------------

AlgArrProc::AlgArrProc(ndarray<const wind_t,2> nda_winds, const unsigned& pbits)
  : m_pbits(pbits)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  //std::cout << "!!! Here in c-tor AlgArrProc(2)\n";
  if(m_pbits & 256) MsgLog(_name(), info, "in c-tor AlgArrProc(2)");

  setWindows(nda_winds);
  setPeakSelectionPars(); // default init

  //v_winds.clear();

  //for( unsigned i=0; i < nda_winds.shape()[0]; ++i) {
  //  const wind_t* p = &nda_winds[i][0];
  //  Window win(p[0], p[1], p[2], p[3], p[4]);
  //  v_winds.push_back(win);
  //}

  if(m_pbits & 1) printInputPars();
}

//--------------------

/*
AlgArrProc::AlgArrProc(const AlgArrProc& obj) 
  : m_pbits(obj.m_pbits)
  , m_is_inited(obj.m_is_inited)
{
  v_winds.clear();
}
*/

//--------------------

/*
AlgArrProc::AlgArrProc(std::vector<Window>& vec_winds, const unsigned& pbits)
  : m_pbits(pbits)
  , m_is_inited(false)
{
  v_winds.clear();

  for(std::vector<Window>::iterator iti = vec_winds.begin(); iti != vec_winds.end(); ++iti) {
    //SegWindow segwin = { iti->segind, iti->window};
    v_winds.push_back(*iti);
  }

  if(m_pbits & 1) printInputPars();
}
*/

//--------------------

AlgArrProc::~AlgArrProc () 
{ 
    if(m_pbits & 256) MsgLog(_name(), info, "in d-tor ~AlgArrProc");
    for(std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) delete *it;
    delete m_mask_def; 
}

//--------------------

void
AlgArrProc::setWindows(ndarray<const wind_t,2> nda_winds)
{
  if(m_pbits & 256) MsgLog(_name(), info, "in setWindows");

  v_winds.clear();

  for( unsigned i=0; i < nda_winds.shape()[0]; ++i) {
    const wind_t* p = &nda_winds[i][0];
    Window win((size_t)p[0], (size_t)p[1], (size_t)p[2], (size_t)p[3], (size_t)p[4]);
    v_winds.push_back(win);
  }
}

//--------------------

void
AlgArrProc::setPeakSelectionPars(const float& npix_min, 
                                 const float& npix_max, 
                                 const float& amax_thr, 
                                 const float& atot_thr,
                                 const float& son_min)
{
  if(m_pbits & 256) MsgLog(_name(), info, "in setPeakSelectionPars");

  m_peak_npix_min = npix_min;
  m_peak_npix_max = npix_max;
  m_peak_amax_thr = amax_thr;
  m_peak_atot_thr = atot_thr;
  m_peak_son_min  = son_min;
}

//--------------------

void 
AlgArrProc::printInputPars()
{
  std::stringstream ss; 
  ss << "printInputPars():"
     << "\n  pbits: " << m_pbits;

  if(v_winds.empty()) { 
    ss << "\n  vector of windows is empty - entire ndarray will be processed.\n";
  } else {
    ss << "\n  Number of windows = " << v_winds.size();
    for(std::vector<Window>::iterator it = v_winds.begin(); it != v_winds.end(); ++ it) ss << '\n' << *it;
  }

  MsgLog(_name(), info, ss.str()); 
}

//--------------------
//--------------------

  const ndarray<const float, 2>
  AlgArrProc::_ndarrayOfPeakPars(const unsigned& npeaks)
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in _ndarrayOfPeakPars, npeaks = " << npeaks);
    if(m_pbits & 1) MsgLog(_name(), info, "List of found peaks, npeaks = " << npeaks); 

    //unsigned shape[2] = {npeaks, 16};
    unsigned sizepk = sizeof(Peak) / sizeof(float);
    //cout << "sizeof(Peak)  = " << sizeof(Peak) << '\n';  // = 64
    //cout << "sizeof(float) = " << sizeof(float) << '\n'; // = 4
    //cout << "num values in Peak = " << sizepk << '\n';   // = 14 (as expected)

    ndarray<float, 2> nda = make_ndarray<float>(npeaks,sizepk);

    unsigned pkcounter = 0;
    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {
      std::vector<Peak>& v_peaks = (*it) -> getVectorOfPeaks();

      for (std::vector<Peak>::iterator ip = v_peaks.begin(); ip != v_peaks.end(); ++ip) {
	const Peak& peak = (*ip);
	float* p_nda = &nda[pkcounter++][0];
        p_nda[ 0] = peak.seg;
        p_nda[ 1] = peak.row;
        p_nda[ 2] = peak.col;
        p_nda[ 3] = peak.npix;       
        p_nda[ 4] = peak.amp_max;    
        p_nda[ 5] = peak.amp_tot;    
        p_nda[ 6] = peak.row_cgrav;   
        p_nda[ 7] = peak.col_cgrav;   
        p_nda[ 8] = peak.row_sigma;  
        p_nda[ 9] = peak.col_sigma;  
        p_nda[10] = peak.row_min;    
        p_nda[11] = peak.row_max;    
        p_nda[12] = peak.col_min;    
        p_nda[13] = peak.col_max;    
        p_nda[14] = peak.bkgd;       
        p_nda[15] = peak.noise;      
        p_nda[16] = peak.son; 

	if(m_pbits & 1) cout << peak << '\n';
      }
    }
    return nda;
  }

//--------------------
//--------------------

  const ndarray<const Peak, 1>
  AlgArrProc::_ndarrayOfPeaks(const unsigned& npeaks)
  {
    if(m_pbits & 256) MsgLog(_name(), info, "in _ndarrayOfPeaks");
    ndarray<Peak, 1> nda = make_ndarray<Peak>(npeaks);

    unsigned pkcounter = 0;
    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {
      std::vector<Peak>& v_peaks = (*it) -> getVectorOfPeaks();

      for (std::vector<Peak>::iterator ip = v_peaks.begin(); ip != v_peaks.end(); ++ip) 
        nda[pkcounter ++] = (*ip);
    }
    return nda;
  }

//--------------------
//--------------------

//template unsigned AlgArrProc::numberOfPixAboveThr<float,2>(const ndarray<const float,2>&, const ndarray<const mask_t,2>&, const float&);

//--------------------
//--------------------
//--------------------

/*
void
AlgArrProc::_initSegmentPars(const unsigned* shape)
{
}
*/
//--------------------
} // namespace ImgAlgos
//--------------------

