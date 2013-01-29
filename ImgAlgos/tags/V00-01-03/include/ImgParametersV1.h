#ifndef IMGALGOS_IMGPARAMETERSV1_H
#define IMGALGOS_IMGPARAMETERSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: ImgParametersV1.h 2726 2012-08-21 14:50:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class ImgParametersV1.
//
//      For 2d image parameters like pedestals, background, gain factor, and mask.
//
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>

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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/**
 *  ImgParametersV1 serves to input and hold data from files for 2d image
 *  per-pixel calibration parameters: pedestals, background, gain factor, mask.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: ImgParametersV1.h 2726 2012-08-21 14:50:00Z dubrovin@SLAC.STANFORD.EDU $
 *
 *  @author Mikhail Dubrovin
 */

class ImgParametersV1  {
public:

  enum { SIZE = 10 }; // Depricated, use it in default constructor only...

  enum { LINESIZEMAX = 2000*10 }; 
  const static char EOL = '\n';

  typedef float pars_t;
  
  // Default constructor
  ImgParametersV1 () {}

  // Constructor, which fills array with value
  ImgParametersV1 (const unsigned* shape, pars_t val=0) ;
  
  // read 2d table of parameters from file use external shape
  ImgParametersV1 (const std::string& fname, const unsigned* shape) ;

  // read 2d table of parameters from file and define the shape from rows, cols in file
  ImgParametersV1 (const std::string& fname) ;

  // Destructor
  ~ImgParametersV1 () ;

  // access parameter data
  ndarray<pars_t, 2> parameters() const {
    return make_ndarray(m_pars, m_rows, m_cols);
  }

  pars_t* data() {return m_pars;}

  void print(std::string comment="");

protected:
  unsigned input_data_from_string(std::string& s);

private:

  // Copy constructor and assignment are disabled by default
  ImgParametersV1 ( const ImgParametersV1& ) ;
  ImgParametersV1& operator = ( const ImgParametersV1& ) ;

  std::vector<pars_t> v_work;

  std::string m_fname;
  pars_t*     m_pars;
  unsigned    m_rows;
  unsigned    m_cols;
  unsigned    m_size;
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGPARAMETERSV1_H
