#ifndef PSCALIB_CALIBPARS_H
#define PSCALIB_CALIBPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
#include <map>
#include <stdint.h> // for uint8_t, uint16_t etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "ndarray/ndarray.h"

//-----------------------------

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
 *  @see CalibFileFinder
 *
 *  Calibration parameters are stored in ndarray<TYPE, NDIM>, where TYPE and NDIM are defined idividually for each type of calibration parameters.
 */

//----------------

enum CALIB_TYPE { PEDESTALS=0, PIXEL_STATUS, PIXEL_RMS, PIXEL_GAIN, PIXEL_MASK, PIXEL_BKGD, COMMON_MODE };

class CalibPars  {
public:

  typedef unsigned shape_t;
  typedef float    pixel_nrms_t;
  typedef float    pixel_bkgd_t;
  typedef uint16_t pixel_mask_t;
  typedef uint16_t pixel_status_t;
  typedef double   common_mode_t;
  typedef float    pedestals_t;
  typedef float    pixel_gain_t;
  typedef float    pixel_rms_t;

  std::map<CALIB_TYPE, std::string> map_type2str;

  // Destructor
  virtual ~CalibPars () {}

  // NOTE1: THE METHOD DECLARED AS
  // virtual ndarray<pedestals_t, 1> pedestals() = 0; IS PURE VIRTUAL,
  // THIS IS NOT OVERLOADABLE BECAUSE THE METHOD SIGNATURE IS DEFINED BY INPUT PARS IN RHS

  // NOTE2: PURE VIRTUAL METHOD NEEDS TO BE IMPLEMENTED IN DERIVED CLASS 
  //        OR IT SHOULD NOT BE "PURE" VIRTUAL, BUT JUST A VIRUAL

  /// Returns number of dimensions in ndarray
  virtual const size_t ndim(const CALIB_TYPE& calibtype=PEDESTALS);

  /// Returns size (number of elements) in calibration type
  virtual const size_t size(const CALIB_TYPE& calibtype=PEDESTALS);

  /// Returns shape of the ndarray with calibration parameters
  virtual const shape_t* shape(const CALIB_TYPE& calibtype=PEDESTALS);

  /// Returns the pointer to array with pedestals 
  virtual const pedestals_t* pedestals();

  /// Returns the pointer to array with pixel_status
  virtual const pixel_status_t* pixel_status();

  /// Returns the pointer to array with pixel_gain 
  virtual const pixel_gain_t* pixel_gain();

  /// Returns the pointer to array with pixel_gain 
  virtual const pixel_rms_t* pixel_rms();

  /// Returns the pointer to array with pixel_mask 
  virtual const pixel_mask_t* pixel_mask();

  /// Returns the pointer to array with pixel_mask 
  virtual const pixel_bkgd_t* pixel_bkgd();

  /// Returns the pointer to array with common_mode 
  virtual const common_mode_t* common_mode();

  /// Partial print of all types of calibration parameters
  virtual void printCalibPars ();

  /// Print map for known calibration types
  void printCalibTypes();

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

protected:

  // Default constructor
  CalibPars () { fill_map_type2str(); }

private:

  void default_msg(const std::string& msg=std::string());

  void fill_map_type2str(); 

};

} // namespace PSCalib

#endif // PSCALIB_CALIBPARS_H
