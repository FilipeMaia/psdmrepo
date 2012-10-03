#ifndef IMGALGOS_CORANAPROCRESULTS_H
#define IMGALGOS_CORANAPROCRESULTS_H

//---------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaProcResults.
//
//---------------------------------

//-----------------
// C/C++ Headers --
//-----------------
//#include <string>
//#include <vector>
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/CorAna.h"
#include "ImgAlgos/TimeInterval.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Merge the input (block vs tau-index) to output (image vs tau-index)
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CorAnaProcResults : public CorAna {
public:

  typedef double hist_t;

  CorAnaProcResults () ;
  virtual ~CorAnaProcResults () ;

protected:

  void readCorFile();
  unsigned getBinInImg(unsigned pix);
  void fillHistogram();
  void saveHistogramInFile();

private:

  cor_t*         m_cor;    // array for correlation results for test

  unsigned*      m_sum0;
  double*        m_sum1;
  hist_t*        m_hist;
  unsigned       m_hsize;
  unsigned       m_nbins;  // Staff for test binning...
  unsigned       m_row_c;  // image center for binning in rings...
  unsigned       m_col_c;  // ... 
  double         m_radmax;
  double         m_radbin;

  TimeInterval*  m_timer1;

  //std::ifstream* p_inp;    // pointer to input array of input files 
  //std::ofstream  p_out;    // pointer to output file 

  // Copy constructor and assignment are disabled by default
  CorAnaProcResults ( const CorAnaProcResults& ) ;
  CorAnaProcResults& operator = ( const CorAnaProcResults& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANAPROCRESULTS_H
