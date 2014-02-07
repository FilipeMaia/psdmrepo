#ifndef PSCALIB_CALIBPARS_H
#define PSCALIB_CALIBPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibPars.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
//#include <vector>
//#include <map>
//#include <fstream>  // open, close etc.

//----------------------
// Base Class Headers --
//----------------------
#include "ndarray/ndarray.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdscalibdata/PnccdPedestalsV1.h"
#include "pdscalibdata/PnccdPixelStatusV1.h"
#include "pdscalibdata/PnccdCommonModeV1.h"
#include "pdscalibdata/PnccdPixelGainV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief Abstract base class CalibPars defining interface to access calibration parameters.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CalibFileFinder
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @endcode
 *
 *  @see CalibFileFinder
 *
 *  Calibration parameters are stored in ndarray<TYPE, NDIM>, where TYPE and NDIM are defined idividually for each type of calibration parameters.
 *
 */

//----------------

class CalibPars  {
public:

  // Destructor
  virtual ~CalibPars () {}

  typedef float    pixel_nrms_t;
  typedef float    pixel_bkgd_t;
  typedef uint16_t pixel_mask_t;

  typedef pdscalibdata::PnccdPixelStatusV1::pars_t pixel_status_t;
  typedef pdscalibdata::PnccdCommonModeV1::pars_t  common_mode_t;
  typedef pdscalibdata::PnccdPedestalsV1::pars_t   pedestals_t;
  typedef pdscalibdata::PnccdPixelGainV1::pars_t   pixel_gain_t;

  //typedef uint16_t pixel_status_t;
  //typedef double   common_mode_t;
  //typedef float    pedestals_t;
  //typedef float    pixel_gain_t;

  // NOTE1: THE METHOD DECLARED AS
  // virtual ndarray<pedestals_t, 1> pedestals() = 0; IS PURE VIRTUAL,
  // THIS IS NOT OVERLOADABLE BECAUSE THE METHOD SIGNATURE IS DEFINED BY INPUT PARS IN RHS

  // NOTE2: PURE VIRTUAL METHOD NEEDS TO BE INPLEMENTED IN DERIVED CLASS 
  //        OR IT SHOULD NOT BE "PURE" VIRTUAL, BUT JUST A VIRUAL

  /// Partial print of all types of calibration parameters
  virtual void printCalibPars      () = 0;
  //virtual void printInputPars      () = 0;
  //virtual void printCalibParsStatus() = 0;
  //virtual int  getCalibTypeStatus(const std::string&  type) = 0;

  /// Returns number of dimensions in ndarray
  virtual const size_t          ndim()         = 0;

  /// Returns size (number of elements) in calibration type
  virtual const size_t          size()         = 0;

  /// Returns shape of the ndarray with calibration parameters
  virtual const unsigned*       shape()        = 0;

  /// Returns the pointer to array with pedestals 
  virtual const pedestals_t*    pedestals()    = 0;

  /// Returns the pointer to array with pixel_status
  virtual const pixel_status_t* pixel_status() = 0;

  /// Returns the pointer to array with common_mode 
  virtual const common_mode_t*  common_mode()  = 0;

  /// Returns the pointer to array with pixel_gain 
  virtual const pixel_gain_t*   pixel_gain()   = 0;

  /*
  virtual ndarray<pedestals_t, 3> pedestals() = 0; 
  virtual void pedestals( ndarray<pedestals_t, 1>& nda ) { nda = make_ndarray<pedestals_t>(2); };
  virtual void pedestals( ndarray<pedestals_t, 2>& nda ) { nda = make_ndarray<pedestals_t>(2, 2); };
  virtual void pedestals( ndarray<pedestals_t, 3>& nda ) { nda = make_ndarray<pedestals_t>(2, 2, 2); };
  virtual void pedestals( ndarray<pedestals_t, 4>& nda ) { nda = make_ndarray<pedestals_t>(2, 2, 2, 2); };

  virtual ndarray<pixel_status_t, 3> pixel_status() = 0;
  virtual void pixel_status( ndarray<pixel_status_t, 1>& nda ) { nda = make_ndarray<pixel_status_t>(2); };
  virtual void pixel_status( ndarray<pixel_status_t, 2>& nda ) { nda = make_ndarray<pixel_status_t>(2, 2); };
  virtual void pixel_status( ndarray<pixel_status_t, 3>& nda ) { nda = make_ndarray<pixel_status_t>(2, 2, 2); };
  virtual void pixel_status( ndarray<pixel_status_t, 4>& nda ) { nda = make_ndarray<pixel_status_t>(2, 2, 2, 2); };

  virtual ndarray<common_mode_t, 1> common_mode() = 0;

  virtual ndarray<pixel_gain_t, 3> pixel_gain() = 0;
  virtual void pixel_gain( ndarray<pixel_gain_t, 1>& nda ) { nda = make_ndarray<pixel_gain_t>(2); };
  virtual void pixel_gain( ndarray<pixel_gain_t, 2>& nda ) { nda = make_ndarray<pixel_gain_t>(2, 2); };
  virtual void pixel_gain( ndarray<pixel_gain_t, 3>& nda ) { nda = make_ndarray<pixel_gain_t>(2, 2, 2); };
  virtual void pixel_gain( ndarray<pixel_gain_t, 4>& nda ) { nda = make_ndarray<pixel_gain_t>(2, 2, 2, 2); };
  */
};

} // namespace PSCalib

#endif // PSCALIB_CALIBPARS_H
