#ifndef IMGALGOS_CORANADATA_H
#define IMGALGOS_CORANADATA_H

//---------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaData.
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
#include "ImgAlgos/ImgVsTimeSplitInFiles.h" // for data_split_t

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
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CorAnaData : public CorAna {
public:

  //typedef uint16_t data_t;
  typedef ImgVsTimeSplitInFiles::data_split_t data_t;

  CorAnaData () ;
  virtual ~CorAnaData () ;

protected:

  void readDataFile();
  void printData();
  void loopProcCorTau();
  void initCorTau();
  void evaluateCorTau(unsigned tau);
  void sumCorTau(unsigned i, unsigned f);
  void saveCorTau(std::ostream& ofs);
  void printCorTau(unsigned tau);

private:

  data_t*     m_data;

  double*     m_sum_gi;
  double*     m_sum_gf;
  double*     m_sum_g2;
  unsigned*   m_sum_st;
  cor_t*      m_cor_gi;
  cor_t*      m_cor_gf;
  cor_t*      m_cor_g2;

  cor_t       m_notzero;

  TimeInterval* m_timer1;

  // Copy constructor and assignment are disabled by default
  CorAnaData ( const CorAnaData& ) ;
  CorAnaData& operator = ( const CorAnaData& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANADATA_H
