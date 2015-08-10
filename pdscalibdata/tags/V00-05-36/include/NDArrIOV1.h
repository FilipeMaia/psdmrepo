
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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"

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
 *  typedef pdscalibdata::NDArrIOV1<float,3> NDAIO;
 *  @endcode
 *
 *  @li Instatiation
 *  \n Constractor 1:
 *  \n Use short name for type and instatiate the object:
 *  @code
 *  std::string fname("path/pedestals/0-end.data"); // mandatory parameter
 *  unsigned shape[]   = {2,3,4};                   // mandatory parameter
 *  TYPE   val_def = 123;                           // optional parameter
 *  unsigned print_bits(0377);                      // optional parameter 
 *
 *  ARRIO* arrio = new ARRIO(fname, shape, data_def, print_bits);
 *  @endcode
 *  where shape is used for 
 *  \n 1) cross-check of metadata shape from file,
 *  \n 2) creation of ndarray<TYPE,NDIM> with default parameters if file is missing.
 *  \n
 *  \n Constractor 2:
 *  @code
 *  CalibPars::common_mode_t data_def[] = {1, 50, 10, Size};
 *  ndarray<CalibPars::common_mode_t,1> nda = make_ndarray(&data_def[0], 4);
 *  ARRIO* arrio = new ARRIO(fname, nda, print_bits);
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *  const ndarray<const float,3>& nda = arrio -> get_ndarray(); // returns ndarray 
 *  // or
 *  const ndarray<const float,3>& nda = arrio -> get_ndarray(fname); // returns ndarray 
 *  std::string& str_status = arrio -> status(); // returns status comment
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
 *  unsigned print_bits=1;
 *  NDAIO::save_ndarray(nda, fname, comments, print_bits);  
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

  static const unsigned c_ndim = NDIM;

public:

  typedef TDATA data_t;
  typedef unsigned shape_t;

  enum STATUS { LOADED=1, DEFAULT, UNREADABLE, UNDEFINED };

  /**
   *  @brief Three constructors provide different default initialization.
   *  Each of them create an object which holds the file name and pointer (0 before load) to ndarray.
   *  File name can be specified later in the get_ndarray(fname) method, but print_file() and print_ndarray()
   *  methods will complain about missing file name until it is specified.
   */ 


  /**
   *  @brief Constructor with missing default initialization. Empty ndarray will be returned if constants from file can not be loaded.
   *  @param[in] fname - std::string file name
   *  @param[in] print_bits - unsigned bit-word to control verbosity
   */ 
  NDArrIOV1 ( const std::string& fname
	    , const unsigned print_bits=0377 );

  /**
   *  @brief Constructor with default ndarray of specified shape filled by a single value.
   *  @param[in] fname - std::string file name
   *  @param[in] shape_def - default shape of the ndarray (is used for shape crosscheck at readout and in case of missing file or metadata)
   *  @param[in] val_def - value to fill all data elements by default(in case of missing file or metadata)
   *  @param[in] print_bits - unsigned bit-word to control verbosity
   */ 
  NDArrIOV1 ( const std::string& fname
	    , const shape_t* shape_def
	    , const TDATA& val_def=TDATA(0) 
	    , const unsigned print_bits=0377 );

  /**
   *  @brief Constructor with externally defined default ndarray.
   *  @param[in] fname - std::string file name
   *  @param[in] nda_def - default ndarray, which will be returned if file is missing 
   *  @param[in] print_bits - unsigned bit-word to control verbosity
   */ 
  NDArrIOV1 ( const std::string& fname
	    , const ndarray<const TDATA, NDIM>& nda_def
	    , const unsigned print_bits=0377 );

  /// Destructor
  ~NDArrIOV1 ();

  /// Returns number of dimensions of ndarray.
  unsigned int ndim() const { return NDIM; }

  /// Access methods
  /// prints recognized templated parameters.
  void print();

  /// Prints input file line-by-line.
  void print_file();

  /// Loads (if necessary) ndarray from file and print it.
  void print_ndarray();

  /// Loads (if necessary) ndarray from file and returns it.
  /**
   *  @param[in] fname std::string file name
   */ 
  ndarray<TDATA, NDIM>& get_ndarray(const std::string& fname = std::string());
  //ndarray<const TDATA, NDIM>& get_ndarray(const std::string& fname = std::string());

  /// Returns string with status of calibration constants.
  std::string str_status();

  /// Returns enumerated status of calibration constants.
  STATUS status() { return m_status; }

  /// Returns string with info about ndarray.
  std::string str_ndarray_info();

  /// Returns string of shape.
  std::string str_shape();

  /// Static method to save ndarray in file with internal metadata and external comments
  /**
   *  @param[in] nda ndarray to save in file
   *  @param[in] fname std::string file name to save ndarray
   *  @param[in] vcoms std::vector<std::string> vector of strings with comments; one-string comment per vector entry
   *  @param[in] print_bits for output control; 0-print nothing, +1-info about saved files
   */ 
  static void save_ndarray(const ndarray<const TDATA, NDIM>& nda, 
                           const std::string& fname,
                           const std::vector<std::string>& vcoms = std::vector<std::string>(), 
	                   const unsigned& print_bits=0377);

protected:

private:

  /// Data members  

  ndarray<TDATA, NDIM>* p_nda;

  std::string m_fname;
  TDATA       m_val_def;
  const ndarray<const TDATA, NDIM> m_nda_def;
  ndarray<TDATA, NDIM> m_nda_empty;
  unsigned    m_print_bits;
  unsigned    m_count_str_data;
  unsigned    m_count_str_comt;
  unsigned    m_count_data;

  unsigned    m_ctor;
  unsigned    m_ndim;
  size_t      m_size;
  shape_t     m_shape[NDIM];
  std::string m_str_type;
  DATA_TYPE   m_enum_type;
  STATUS      m_status;

  TDATA* p_data;

  /// static method returns class name for MsgLog
  static std::string __name__(){ return std::string("NDArrIOV1"); }

  /// member data common initialization in constructors
  void init();

  /// loads metadata and ndarray<TYPE,NDIM> from file
  void load_ndarray();

  /// true if the file name non empty anf file is readable
  bool file_is_available();

  /// parser for comment lines and metadata from file with ndarray
  /**
   *  @param[in] str one string of comments from file
   */ 
  void parse_str_of_comment(const std::string& str);

  /// creates ndarray, begins to fill data from the 1st string and reads data by the end
  /**
   *  @param[in] in input file stream
   *  @param[in] str 1st string of the data
   */ 
  void load_data(std::ifstream& in, const std::string& str);

  /// creates ndarray<TYPE,NDIM> with shape from constructor parameter or metadata.
  /**
   *  @param[in] fill_def if true - fills ndarray with default values
   */ 
  void create_ndarray(const bool& fill_def=false);

  /// Copy constructor and assignment are disabled by default
  NDArrIOV1 ( const NDArrIOV1& ) ;
  NDArrIOV1& operator = ( const NDArrIOV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_NDARRIOV1_H
