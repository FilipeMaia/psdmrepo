//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AlgDroplet
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

#include "ImgAlgos/AlgDroplet.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>    // for exp
#include <iomanip>   // for std::setw
#include <sstream>   // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {


//----------------
// Constructors --
//----------------
//template <typename T>
//AlgDroplet<T>::AlgDroplet(const double& sigma, const int& nsm) 

AlgDroplet::AlgDroplet(   const int&      radius
                        , const double&   thr_low
                        , const double&   thr_high
                        , const unsigned& pbits
                        , const size_t&   seg
                        , const size_t&   rowmin
                        , const size_t&   rowmax
                        , const size_t&   colmin
                        , const size_t&   colmax
                        )
  : m_radius(radius)
  , m_thr_low(thr_low)
  , m_thr_high(thr_high)
  , m_pbits(pbits)
  , m_seg(seg)
  , m_rowmin(rowmin)
  , m_rowmax(rowmax)
  , m_colmin(colmin)
  , m_colmax(colmax)
{
  if(m_pbits & 1) printInputPars();
}

//--------------------
// Print member data and matrix of weights
void 
AlgDroplet::printInputPars()
{
  std::stringstream ss; 
  ss << "radius  : " << m_radius
     << "thr_low : " << m_thr_low
     << "thr_high: " << m_thr_high
     << "pbits   : " << m_pbits
     << "seg     : " << m_seg
     << "rowmin  : " << m_rowmin
     << "rowmax  : " << m_rowmax
     << "colmin  : " << m_colmin
     << "colmax  : " << m_colmax  
     << '\n';
  MsgLog(name(), info, ss.str()); 
}

//--------------------
// Save droplet info in vector
void 
AlgDroplet::saveDropletInfo(size_t& row, size_t& col, double& amp, double& amp_tot, unsigned& npix )
{
  if ( v_droplets.size() == v_droplets.capacity() ) {
      v_droplets.reserve( v_droplets.capacity() + NDROPLETSBLOCK );
      if(m_pbits & 4) MsgLog( name(), info, "Droplets vector capacity is increased to:" << v_droplets.capacity() );
  }
  Droplet oneDroplet = { (double)col, (double)row, amp, amp_tot, npix };
  v_droplets.push_back(oneDroplet);
  if(m_pbits & 4) printDropletInfo( oneDroplet );
}

//--------------------
// Print droplet info
void 
AlgDroplet::printDropletInfo(const Droplet& d)
{
  MsgLog(name(), info, "Droplet is found: "  << strDropletPars(d) 
                       << " v_droplets.size()=" << v_droplets.size()
                       << " capacity()=" << v_droplets.capacity() );
}

//--------------------
// Returns string with droplet parameters
std::string
AlgDroplet::strDropletPars(const Droplet& d)
{
  std::stringstream ss; 
  ss <<  "x="      << d.x 
     << " y="      << d.y
     << " ampmax=" << d.ampmax
     << " amptot=" << d.amptot 
     << " npix="   << d.npix;
  return ss.str();  
}

//--------------------
// Print all droplets
void 
AlgDroplet::printDroplets()
{
  std::stringstream ss; ss << "Vector of Droplets"
                           << " v_droplets.size()=" << v_droplets.size()
                           << " capacity()=" << v_droplets.capacity();

  for( vector<Droplet>::iterator itv  = v_droplets.begin();
                                 itv != v_droplets.end(); itv++ ) {
    
    ss <<  "   " << strDropletPars(*itv) << '\n';  
  }
  MsgLog(name(), info, ss.str()); 
}

//--------------------
//--------------------

//template class ImgAlgos::AlgDroplet<int16_t>;
//template class ImgAlgos::AlgDroplet<int>;
//template class ImgAlgos::AlgDroplet<float>;
//template class ImgAlgos::AlgDroplet<double>;

//--------------------

} // namespace ImgAlgos
