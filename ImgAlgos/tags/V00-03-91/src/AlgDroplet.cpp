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
  std::stringstream ss; ss << "Input parameters:";
  ss << "\nradius  : " << m_radius
     << "\nthr_low : " << m_thr_low
     << "\nthr_high: " << m_thr_high
     << "\npbits   : " << m_pbits
     << "\nseg     : " << m_seg
     << "\nrowmin  : " << m_rowmin
     << "\nrowmax  : " << m_rowmax
     << "\ncolmin  : " << m_colmin
     << "\ncolmax  : " << m_colmax;
  MsgLog(_name(), info, ss.str()); 
}

//--------------------
// Save droplet info in vector
void 
AlgDroplet::_saveDropletInfo(size_t& seg, size_t& row, size_t& col, double& amp, double& amp_tot, unsigned& npix )
{
  if( v_droplets.size() == v_droplets.capacity() ) {
      v_droplets.reserve( v_droplets.capacity() + NDROPLETSBLOCK );
      if(m_pbits & 2) MsgLog( _name(), info, "Droplets vector capacity is increased to:" << v_droplets.capacity() );
  }
  Droplet oneDroplet = {(unsigned)seg, (double)row, (double)col, amp, amp_tot, npix};
  v_droplets.push_back(oneDroplet);
  //if(m_pbits & 32) _printDropletInfo( oneDroplet );
}

//--------------------
// Print droplet info
void 
AlgDroplet::_printDropletInfo(const Droplet& d)
{
  MsgLog(_name(), info, "Droplet is found: "  << _strDropletPars(d) 
                       << " v_droplets.size()=" << v_droplets.size()
                       << " capacity()=" << v_droplets.capacity() );
}

//--------------------
// Returns string with droplet parameters
std::string
AlgDroplet::_strDropletPars(const Droplet& d)
{
  std::stringstream ss; 
  ss << "  seg="    << std::setw(4) << d.seg 
     << "  row="    << std::setw(4) << d.row 
     << "  col="    << std::setw(4) << d.col
     << "  ampmax=" << std::fixed << std::setprecision(2) << std::setw(8) << d.ampmax
     << "  amptot=" << std::fixed << std::setprecision(2) << std::setw(8) << d.amptot 
     << "  npix="   << std::setw(4) << d.npix;
  return ss.str();  
}

//--------------------
// Print all droplets
void 
AlgDroplet::printDroplets()
{
  std::stringstream ss; ss << "Vector of Droplets v_droplets in seg:" << m_seg
                           << " size:" << v_droplets.size()
                           << " capacity:" << v_droplets.capacity()
                           << '\n';  

  for( vector<Droplet>::iterator itv  = v_droplets.begin();
                                 itv != v_droplets.end(); itv++ ) {
    
    ss <<  "   " << _strDropletPars(*itv) << '\n';  
  }
  MsgLog(_name(), info, ss.str()); 
}

//--------------------

//template class ImgAlgos::AlgDroplet<int16_t>;
//template class ImgAlgos::AlgDroplet<int>;
//template class ImgAlgos::AlgDroplet<float>;
//template class ImgAlgos::AlgDroplet<double>;

//--------------------

} // namespace ImgAlgos
