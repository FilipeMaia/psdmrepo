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
#include <sstream>   // for stringstream
#include <cstring>  // for memcpy

namespace ImgAlgos {

  typedef AlgArrProc::wind_t wind_t;
  typedef AlgImgProc::conmap_t conmap_t;

//--------------------
// Constructors --
//--------------------

/*
AlgArrProc::AlgArrProc()
  : m_pbits(0)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  if(m_pbits & 256) MsgLog(_name(), info, "in c-tor:0 AlgArrProc()");

  default init
  v_winds.clear();
  setSoNPars();
  setPeakSelectionPars();
}
*/

//--------------------

AlgArrProc::AlgArrProc(const unsigned& pbits)
  : m_pbits(pbits)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  if(m_pbits & 256) MsgLog(_name(), info, "in c-tor:1 AlgArrProc(.)");

  // default init
  v_winds.clear();
  setSoNPars();
  setPeakSelectionPars();
}

//--------------------

AlgArrProc::AlgArrProc(ndarray<const wind_t,2> nda_winds, const unsigned& pbits)
  : m_pbits(pbits)
  , m_is_inited(false)
  , m_mask_def(0)
  , m_mask(0)
{
  if(m_pbits & 256) MsgLog(_name(), info, "in c-tor:2 AlgArrProc(..)");

  setWindows(nda_winds);
  // default init
  setSoNPars();
  setPeakSelectionPars();

  if(m_pbits & 2) printInputPars();
}

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
AlgArrProc::setSoNPars(const float& r0, const float& dr)
{ 
  if(m_pbits & 256) MsgLog(_name(), info, "in setSoNPars, r0=" << r0 << " dr=" << dr);

  m_r0=r0; 
  m_dr=dr; 
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
  ss << "\n  S/N evaluation parameters:"
     << "\n    r0: " << m_r0
     << "\n    dr: " << m_dr;

  ss << "\n  Peak selection parameters:"
     << "\n    npix_min: " << m_peak_npix_min
     << "\n    npix_max: " << m_peak_npix_max
     << "\n    amax_thr: " << m_peak_amax_thr
     << "\n    atot_thr: " << m_peak_atot_thr
     << "\n    son_min : " << m_peak_son_min;

  MsgLog(_name(), info, ss.str()); 

  if(v_algip.size()>0) {
    for(std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) (*it) -> printInputPars();
  }
  else MsgLog(_name(), info, "v_algip is empty...");
}

//--------------------

const ndarray<const float, 2>
AlgArrProc::_ndarrayOfPeakPars(const unsigned& npeaks)
{
    if(m_pbits & 256) MsgLog(_name(), info, "in _ndarrayOfPeakPars, npeaks = " << npeaks);
    if(m_pbits & 1) MsgLog(_name(), info, "List of found peaks, npeaks = " << npeaks); 

    unsigned sizepk = sizeof(Peak) / sizeof(float);

    ndarray<float, 2> nda = make_ndarray<float>(npeaks,sizepk);

    unsigned pkcounter = 0;
    for (std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {
      //std::vector<Peak>& peaks = (*it) -> getVectorOfPeaks();
      std::vector<Peak>& peaks = (*it) -> getVectorOfSelectedPeaks();

      for (std::vector<Peak>::iterator ip = peaks.begin(); ip != peaks.end(); ++ip) {
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

ndarray<const conmap_t, 3>
AlgArrProc::mapsOfConnectedPixels()
{
  if(m_pbits & 256) MsgLog(_name(), info, "in mapsOfConnectedPixels");

  unsigned shape[3] = {m_nsegs, m_nrows, m_ncols};
  ndarray<conmap_t, 3> maps(shape);

  for(std::vector<AlgImgProc*>::iterator it = v_algip.begin(); it != v_algip.end(); ++it) {

    ndarray<conmap_t, 2>& conmap = (*it) -> mapOfConnectedPixels();
    const Window& win = (*it) -> window();

    for(unsigned r = win.rowmin; r<win.rowmax; r++) 
      for(unsigned c = win.colmin; c<win.colmax; c++)
        maps[win.segind][r][c] = conmap[r][c];
  }
  return maps;
}

//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
//--------------------
//--------------------

