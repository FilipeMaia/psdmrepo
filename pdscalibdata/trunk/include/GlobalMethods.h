#ifndef PDSCALIBDATA_GLOBALMETHODS_H
#define PDSCALIBDATA_GLOBALMETHODS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <fstream>   // ofstream
#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
#include <iostream>
#include <stdexcept>
#include <typeinfo>  // for typeid
#include <stdint.h>    // for uint8_t, uint16_t etc.
//#include <algorithm>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/// @addtogroup pdscalibdata

/**
 *  @ingroup pdscalibdata
 *
 *  @brief Global methods for pdscalibdata package
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

using namespace std;
 
enum DATA_TYPE {FLOAT, DOUBLE, SHORT, UNSIGNED, INT, INT16, INT32, UINT, UINT8, UINT16, UINT32};

const static int UnknownCM = -10000; 

class GlobalMethods  {
public:
  GlobalMethods () ;
  virtual ~GlobalMethods () ;

private:
  // Copy constructor and assignment are disabled by default
  GlobalMethods ( const GlobalMethods& ) ;
  GlobalMethods& operator = ( const GlobalMethods& ) ;
};

//--------------------

  std::string stringFromUint(unsigned number, unsigned width=6, char fillchar='0');

  void printSizeOfTypes();

//--------------------

  /**
   *  Find common mode for an CsPad  section.
   *  
   *  Function will return UnknownCM value if the calculation 
   *  cannot be performed (or need not be performed).
   *  
   *  @param pars   array[3] of control parameters
   *  @param sdata  pixel data
   *  @param peddata  pedestal data, can be zero pointer
   *  @param pixStatus  pixel status data, can be zero pointer
   *  @param ssize  size of all above arrays
   *  @param stride increment for pixel indices
   */ 
  float findCommonMode(const double* pars,
                       const int16_t* sdata,
                       const float* peddata, 
                       const  uint16_t *pixStatus, 
                       unsigned ssize,
                       int stride = 1); // const;
  
//--------------------
// For type=T returns the string with symbolic data type and its size, i.e. "d of size 8"
  template <typename T>
  std::string strOfDataTypeAndSize()
  {
    std::stringstream ss; ss << typeid(T).name() << " of size " << sizeof(T);
    return ss.str();
  }

//--------------------
  /**
   * @brief Load parameters from file
   * 
   * @param[in]  fname - path to the file with parameters
   * @param[in]  comment - comment to print with messages 
   * @param[in]  size  - size of array with parameters; number of values to load from file
   * @param[out] pars  - pointer to array with parameters
   * @param[in]  check_bits - +1: check if input does not have enough data, +2: check if file has extra data 
   */

  template <typename T>
  void load_pars_from_file(const std::string& fname, const std::string& comment, const size_t size, T* pars, const unsigned check_bits=255)
  { 
    // open file
    std::ifstream in(fname.c_str());
    if (not in.good()) {
      const std::string msg = "Failed to open pedestals file: "+fname;
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }
  
    // read all numbers
    T* it = pars;
    size_t count = 0;
    while(in and count != size) {
      in >> *it++;
      ++ count;
    }
  
    // check that we read whole array
    if (check_bits & 1 && count < size) {
      const std::string msg = comment+" file does not have enough data: "+fname;
      MsgLogRoot(error, msg);
      throw std::runtime_error(msg);
    }
  
    // and no data left after we finished reading
    float tmp ;
    if (check_bits & 2 && in >> tmp ) {
      ++ count;
      const std::string msg = comment+" file has extra data: "+fname;
      MsgLogRoot(error, msg);
      MsgLogRoot(error, "read " << count << " numbers, expecting " << size );
      throw std::runtime_error(msg);
    }
  }

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

} // namespace pdscalibdata

#endif // PDSCALIBDATA_GLOBALMETHODS_H
