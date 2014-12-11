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
                        , const size_t&   seg
                        , const size_t& rowmin
                        , const size_t& rowmax
                        , const size_t& colmin
                        , const size_t& colmax
                        )
  : m_sigma(sigma)
  , m_nsm(nsm)
  , m_nsm1(nsm+1)
  , m_thr_low(thr_low)
  , m_opt(opt)
  , m_seg(seg)
  , m_rowmin(rowmin)
  , m_rowmax(rowmax)
  , m_colmin(colmin)
  , m_colmax(colmax)
{
  evaluateWeights();
}

//--------------------
// Define smearing weighting matrix
void 
AlgSmearing::evaluateWeights()
{
  m_weights = make_ndarray<double>(m_nsm1, m_nsm1);

  if ( m_sigma == 0 ) { 
    MsgLog( name(), info, "Smearing is turned OFF by sigma = " << m_sigma ); 
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
  // if ( m_sigma == 0 ) { MsgLog( name(), info, "Smearing is turned OFF by sigma =" << m_sigma ); }

  WithMsgLog(name(), info, log) { log << "printWeights() - Weights for smearing";
    for (int r = 0; r < m_nsm1; r++) {
          log << "\n   row=" << r << ":     "; 
      for (int c = 0; c < m_nsm1; c++) {
          std::stringstream ss; ss.setf(ios::fixed,ios::floatfield); //setf(ios::left);
          ss << std::left << std::setw(8) << m_weights[r][c];
          log << ss.str() << "  ";       
      }
    }
  } // WithMsgLog 
}

//--------------------
// Print member data and matrix of weights
void 
AlgSmearing::printInputPars()
{
  std::stringstream ss; 
  ss << "sigma   : " << m_sigma
     << "nsm     : " << m_nsm
     << "nsm1    : " << m_nsm1
     << "thr_low : " << m_thr_low
     << "opt     : " << m_opt
     << "seg     : " << m_seg
     << "rowmin  : " << m_rowmin
     << "rowmax  : " << m_rowmax
     << "colmin  : " << m_colmin
     << "colmax  : " << m_colmax  
     << '\n';
  MsgLog(name(), info, ss.str()); 
}

//--------------------

//template class ImgAlgos::AlgSmearing<int16_t>;
//template class ImgAlgos::AlgSmearing<int>;
//template class ImgAlgos::AlgSmearing<float>;
//template class ImgAlgos::AlgSmearing<double>;

//--------------------

} // namespace ImgAlgos
