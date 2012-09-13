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
#include <string>
#include <vector>
#include <fstream>  // for ostream, ofstream
#include <iostream> // for cout, puts etc.

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CorAnaData  {
public:

  typedef uint16_t data_t;

  CorAnaData () ;
  virtual ~CorAnaData () ;

protected:

  void readMetadataFile();
  void printMetadata();
  void readDataFile();
  void printData();
  void loopProcCorTau();
  void initCorTau();
  void evaluateCorTau(unsigned tau);
  void sumCorTau(unsigned i, unsigned f);
  void saveCorTau(unsigned tau);
  void printCorTau(unsigned tau);
  void readIndTauFromFile();
  void makeIndTau();
  void printIndTau();
  void saveIndTauInFile();

private:

  std::ostream& m_log;

  std::string  m_fname;
  std::string  m_fname_com;
  std::string  m_fname_med;
  std::string  m_fname_tau;
  std::string  m_fname_tau_out;

  std::string  m_file_type;
  std::string  m_data_type;

  unsigned    m_img_rows;
  unsigned    m_img_cols;
  unsigned    m_img_size;
  unsigned    m_nfiles;
  unsigned    m_blk_size;
  unsigned    m_rst_size;
  unsigned    m_nimgs;

  data_t*     m_data;

  double*     m_res_g2;
  double*     m_sum_g2;
  double*     m_sum_gi;
  double*     m_sum_gf;
  unsigned*   m_sum_st;

  vector<unsigned> v_ind_tau;

  TimeInterval* m_timer1;

  // Copy constructor and assignment are disabled by default
  CorAnaData ( const CorAnaData& ) ;
  CorAnaData& operator = ( const CorAnaData& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANADATA_H
