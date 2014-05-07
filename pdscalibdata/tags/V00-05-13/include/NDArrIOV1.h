
#ifndef PDSCALIBDATA_NDARRIOV1_H
#define PDSCALIBDATA_NDARRIOV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//      $Revision$
//      $HeadURL$
//      $Header$
//      $LastChangedDate$
//      $Date$
//
// Author: Mikhail Dubrovin
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <iostream> // for cout, puts etc.

//----------------------
// Base Class Headers --
//----------------------
//#include "pdscalibdata/PnccdBaseV1.h" // Segs, Rows, Cols etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
//#include "pdsdata/psddl/pnccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdscalibdata/GlobalMethods.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

using namespace std;

namespace pdscalibdata {

/**
 *  @defgroup pdscalibdata pdscalibdata package
 *  @brief Package pdscalibdata contains modules which retrieves the calibration parameters of all detectors
 */

/**
 *  @ingroup pdscalibdata
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 *
 *
 *  @anchor interface
 *  @par<interface> Interface
 *
 *  @li Expected format of the data file with metadata
 *  @code
 *  # line of comment always begins with # 
 *  # Mandatory fields to define the ndarray<TYPE,NDIM> and its shape as unsigned shape[NDIM] = {DIM:1,DIM:2,DIM:3}
 *  # TYPE        float
 *  # NDIM        3
 *  # DIM:1       3
 *  # DIM:2       4
 *  # DIM:3       8
 * 
 *  21757   21769   33464   10769   68489   68561   77637   54810 
 *  ... other data lines
 *  21757   21773   33430   10628   68349   68345   77454   54729 
 *  @endcode
 *
 *  @li Includes and typedefs
 *  @code
 *  #include "pdscalibdata/NDArrIOV1.h"
 *  typedef pdscalibdata::NDArrIOV1 ARRIO;
 *  @endcode
 *
 *  @li Instatiation
 *  \n Use short name for type and instatiate the object:
 *  @code
 *  ARRIO* arrio = ARRIO();
 *  // or
 *  std::string fname("path/pedestals/0-end.data");
 *  ARRIO* arrio = ARRIO(fname);
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *  const ndarray<const float,3>& nda = arrio -> get_ndarray(); // returns ndarray 
 *  // or
 *  const ndarray<const float,3>& nda = arrio -> get_ndarray(fname); // returns ndarray 
 *  @endcode
 *
 *  @li Print methods
 *  \n File name needs to be provided either in constructor or in the get_ndarray(fname) method.
 *  @code
 *  arrio -> print();         // prints recognized templated parameters
 *  arrio -> print_file();    // prints input file line-by-line 
 *  arrio -> print_ndarray(); // prints ndarray 
 *  @endcode
 *
 *  @li Save ndarray in file
 *  @code
 *  NDAIO::save_ndarray(nda, fname);  
 *  @endcode
 *  or save ndarray in file with additional comments
 *  @code
 *  std::vector<std::string> comments;
 *  comments.push_back("TITLE      File to load ndarray of calibration parameters");
 *  comments.push_back("EXPERIMENT amo12345");
 *  comments.push_back("DETECTOR   Camp.0:pnCCD.1");
 *  comments.push_back("CALIB_TYPE pedestals");
 *
 *  NDAIO::save_ndarray(nda, fname, comments);  
 *  @endcode
 *  where each record in the vector is added to the file header as a commented string. 
 *  The last command saves the file with content
 *  @code
 *  # TITLE      File to load ndarray of calibration parameters
 *  # 
 *  # EXPERIMENT amo12345
 *  # DETECTOR   Camp.0:pnCCD.1
 *  # CALIB_TYPE pedestals
 *  
 *  # DATE_TIME  2014-05-06 15:24:10
 *  # AUTHOR     <user-login-name>
 *  
 *  # Metadata for ndarray<float,3>
 *  # TYPE     float
 *  # NDIM     3
 *  # DIM:0    3
 *  # DIM:1    4
 *  # DIM:2    8
 *  
 *       21757      21769      33464      10769      68489      68561      77637      54810 
 *       ...
 *         102         40        272        270        194        246         60        118 
 *  @endcode
 */

template <typename TDATA, unsigned NDIM> // stands for something like ndarray<TDATA, NDIM>
class NDArrIOV1 {

  static const unsigned int c_ndim = NDIM;
  typedef TDATA data_t;

  typedef unsigned int shape_t;


public:
  /// Constructor with/without file name
  /**
   *  @brief creates an object which holds the file name and pointer (0 before load) to ndarray.
   *  File name can be passed later in the get_ndarray(fname) method, but print_file() and print_ndarray() 
   *  methods will complain about missing file name until it is specified.
   *  @param[in] fname std::string file name
   *  @param[in] print_bits unsigned bit-word to control verbosity
   */ 
  NDArrIOV1 (const std::string& fname = std::string(), unsigned print_bits=0377);

  /// Destructor
  ~NDArrIOV1 (){}

  unsigned int ndim() const { return NDIM; }

  /// Access methods
  /// prints recognized templated parameters
  void print();

  /// prints input file line-by-line 
  void print_file();

  /// load (if necessary) ndarray from file and print it
  void print_ndarray();

  /// load (if necessary) ndarray from file and returns it
  /**
   *  @param[in] fname std::string file name
   */ 
  ndarray<const TDATA, NDIM> get_ndarray(const std::string& fname = std::string());

  /// Save ndarray in file with metadata internal and external comments
  /**
   *  @param[in] nda ndarray to save in file
   *  @param[in] fname std::string file name to save ndarray
   *  @param[in] vcoms std::vector<std::string> vector of strings with comments
   */ 
  static void save_ndarray(const ndarray<const TDATA, NDIM>& nda, 
                           const std::string& fname,
                           const std::vector<std::string>& vcoms = std::vector<std::string>());

protected:

private:

  /// Data members  

  std::string m_fname;
  unsigned    m_print_bits;
  unsigned    m_count_str_data;
  unsigned    m_count_str_comt;
  unsigned    m_count_data;

  unsigned    m_ndim;
  size_t      m_size;
  shape_t     m_shape[NDIM];
  std::string m_str_type;
  DATA_TYPE   m_enum_type;

  ndarray<TDATA, NDIM>* p_nda;
  TDATA* p_data;

  static std::string __name__(){ return std::string("NDArrIOV1"); }
  bool file_is_available();
  void parse_str_of_comment(std::string str);
  void parse_str_of_data(std::string str);
  void create_ndarray();
  void check_input_consistency();
  /// loads metadata and ndarray<TYPE,NDIM> from file
  void load_ndarray();
  
  /// Copy constructor and assignment are disabled by default
  NDArrIOV1 ( const NDArrIOV1& ) ;
  NDArrIOV1& operator = ( const NDArrIOV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_NDARRIOV1_H
