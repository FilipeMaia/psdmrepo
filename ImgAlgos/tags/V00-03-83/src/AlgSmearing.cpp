//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AlgSmearing
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AlgSmearing.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>    // for exp
#include <iomanip>   // for std::setw
#include <sstream>   // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

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
//AlgSmearing<T>::AlgSmearing(const double& sigma, const int& nsm) 

AlgSmearing::AlgSmearing( const double& sigma
                        , const int& nsm
                        , const double& thr_low
                        , const unsigned& opt
                        , const unsigned& pbits
                        , const size_t&   seg
                        , const size_t& rowmin
                        , const size_t& rowmax
                        , const size_t& colmin
                        , const size_t& colmax
                        , const double& value
                        )
  : m_sigma(sigma)
  , m_nsm(nsm)
  , m_nsm1(nsm+1)
  , m_thr_low(thr_low)
  , m_opt(opt)
  , m_pbits(pbits)
  , m_seg(seg)
  , m_rowmin(rowmin)
  , m_rowmax(rowmax)
  , m_colmin(colmin)
  , m_colmax(colmax)
  , m_value(value)
{
  if(m_pbits & 1) printInputPars();
  evaluateWeights();
  if(m_pbits & 2) printWeights();
}

//--------------------
// Define smearing weighting matrix
void 
AlgSmearing::evaluateWeights()
{
  m_weights = make_ndarray<double>(m_nsm1, m_nsm1);

  //std::cout << "In AlgSmearing::evaluateWeights()\n";

  if ( m_sigma == 0 ) { 
    if(m_pbits) MsgLog(_name(), info, "Smearing is turned OFF by sigma = " << m_sigma); 
    std::fill_n(m_weights.data(), int(m_weights.size()*sizeof(double)), double(0));
    m_weights[0][0] = 1;
    return;
  }

  double norm = -0.5/(m_sigma*m_sigma);
  for (int r = 0; r < m_nsm1; r++) {
    for (int c = 0; c < m_nsm1; c++) {
      m_weights[r][c] = exp( norm * (r*r+c*c) );
    }
  }
}

//--------------------
// Print smearing weighting matrix
void 
AlgSmearing::printWeights()
{
  // if ( m_sigma == 0 ) { MsgLog(_name(), info, "Smearing is turned OFF by sigma =" << m_sigma ); }

    std::stringstream ss; ss << "printWeights() - Weights for smearing";
    ss << "\n   cols :   ";
    for (int c = 0; c < m_nsm1; c++) ss << std::left << std::setw(10) << c;
    for (int r = 0; r < m_nsm1; r++) {
      ss << "\n   row=" << r << ": " << fixed; 
      for (int c = 0; c < m_nsm1; c++) {
	ss << "  " << std::left << std::setw(8) << m_weights[r][c];
      }
    }
    MsgLog(_name(), info, ss.str()); 
}

//--------------------
// Print member data and matrix of weights
void 
AlgSmearing::printInputPars()
{
  std::stringstream ss; 
  ss << "\nsigma   : " << m_sigma
     << "\nnsm     : " << m_nsm
     << "\nnsm1    : " << m_nsm1
     << "\nthr_low : " << m_thr_low
     << "\nopt     : " << m_opt
     << "\nseg     : " << m_seg
     << "\nrowmin  : " << m_rowmin
     << "\nrowmax  : " << m_rowmax
     << "\ncolmin  : " << m_colmin
     << "\ncolmax  : " << m_colmax  
     << "\nvalue   : " << m_value  
     << '\n';
  MsgLog(_name(), info, ss.str()); 
}

//--------------------

//template class ImgAlgos::AlgSmearing<int16_t>;
//template class ImgAlgos::AlgSmearing<int>;
//template class ImgAlgos::AlgSmearing<float>;
//template class ImgAlgos::AlgSmearing<double>;

//--------------------

} // namespace ImgAlgos
