#ifndef IMGALGOS_CORANA_H
#define IMGALGOS_CORANA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAna.
//
//------------------------------------------------------------------------

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
 *  This superclass module contains common I/O infrastructure for correlation analysis
 *
 *
 *
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

struct TimeRecord {
  unsigned    evind;
  double      t_sec;
  double      dt_sec;
  std::string tstamp;
  unsigned    fiduc;
  unsigned    evnum;
  unsigned    tind;
};


class CorAna {
public:

  typedef float cor_t;

  // Default constructor
  CorAna () ;

  // Destructor
  virtual ~CorAna () ;

protected:

  std::ostream& m_log;

  std::string  m_fname;
  std::string  m_fname_com;
  std::string  m_fname_med;
  std::string  m_fname_time;
  std::string  m_fname_time_ind;
  std::string  m_fname_tau;
  std::string  m_fname_tau_out;
  std::string  m_fname_result;
  std::string  m_fname_result_img;
  std::string  m_fname_hist;
  std::string  m_file_num_str;

  unsigned     m_img_rows;
  unsigned     m_img_cols;
  unsigned     m_img_size;
  unsigned     m_nfiles;
  unsigned     m_blk_size;
  unsigned     m_rst_size;
  unsigned     m_nimgs;
  std::string  m_file_type;
  std::string  m_data_type;
  std::string  m_data_type_input;
  double       m_t_ave;   
  double       m_t_rms;   
  unsigned     m_tind_max;
  unsigned     m_tind_size;

  int*         m_tind_to_evind;

  std::vector<TimeRecord> v_time_records;

  std::vector<unsigned> v_ind_tau;
  unsigned    m_npoints_tau;

  void defineFileNames();
  void printFileNames();
  void readMetadataFile();
  void printMetadata();
  void readTimeRecordsFile();
  void printTimeRecords();
  void printTimeIndexToEventIndexArr();
  void defineIndTau();
  int  readIndTauFromFile();
  void makeIndTau();
  void printIndTau();
  void saveIndTauInFile();

private:

  // Copy constructor and assignment are disabled by default
  CorAna ( const CorAna& ) ;
  CorAna& operator = ( const CorAna& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANA_H
