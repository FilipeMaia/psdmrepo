#ifndef PSCALIB_SEGGEOMETRYSTORE_H
#define PSCALIB_SEGGEOMETRYSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
//#include <iostream>
#include <string>

//#include "PSCalib/SegGeometry.h"
#include "PSCalib/SegGeometryCspad2x1V1.h"
//#include "PSCalib/SegGeometryCspad2x1V2.h"
//#include "PSCalib/SegGeometryCspad2x1V3.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "ImgAlgos/GlobalMethods.h" // ::toString( const Pds::Src& src )
//#include "ndarray/ndarray.h"
//#include "pdsdata/xtc/Src.hh"
#include "MsgLogger/MsgLogger.h"

//-----------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief class SegGeometryStore has a static factory method Create for SegGeometry object
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see SegGeometry, SegGeometryCspad2x1V1
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Includes
 *  @code
 *  // #include "PSCalib/SegGeometry.h" // already included under SegGeometryStore.h
 *  #include "PSCalib/SegGeometryStore.h"
 *  typedef PSCalib::SegGeometry SG;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Classes like SegGeometryCspad2x1V1 containing implementation of the SegGeometry interface methods are self-sufficient. 
 *  Factory method Create should returns the pointer to the SegGeometry object by specified segname parameter or returns 0-pointer if segname is not recognized (and not implemented).
 *  Code below instateates SegGeometry object using factory static method PSCalib::SegGeometryStore::Create()
 *  @code
 *  std::string source = "SENS2X1:V1";
 *  unsigned print_bits=0377; // does not print by default if parameter omited
 *  PSCalib::SegGeometry* segeo = PSCalib::SegGeometryStore::Create(segname, print_bits);
 *  @endcode
 *
 *  @li Print info
 *  @code
 *  segeo -> print_seg_info(0377);
 *  @endcode
 *
 *  @li Access methods
 *  \n are defined in the interface SegGeometry and implemented in SegGeometryCspad2x1V1
 *  @code
 *  // scalar values
 *  const SG::size_t         array_size        = segeo -> size(); 
 *  const SG::size_t         number_of_rows    = segeo -> rows();
 *  const SG::size_t         number_of_cols    = segeo -> cols();
 *  const SG::pixel_coord_t  pixel_scale_size  = segeo -> pixel_scale_size();
 *  const SG::pixel_coord_t  pixel_coord_min   = segeo -> pixel_coord_min(SG::AXIS_Z);
 *  const SG::pixel_coord_t  pixel_coord_max   = segeo -> pixel_coord_max(SG::AXIS_X);
 * 
 *  // pointer to arrays with size equal to array_size
 *  const SG::size_t*        p_array_shape     = segeo -> shape();
 *  const SG::pixel_area_t*  p_pixel_area      = segeo -> pixel_area_array();
 *  const SG::pixel_coord_t* p_pixel_size_arr  = segeo -> pixel_size_array(SG::AXIS_X);
 *  const SG::pixel_coord_t* p_pixel_coord_arr = segeo -> pixel_coord_array(SG::AXIS_Y);
 *  @endcode
 *
 *  @li How to add new segment to the factory
 *  \n 1. implement SegGeometry interface methods in class like SegGeometryCspad2x1V1
 *  \n 2. add it to SegGeometryStore with unique segname
 */

//----------------

class SegGeometryStore  {
public:

  //SegGeometryStore () {}
  //virtual ~SegGeometryStore () {}

  /**
   *  @brief Static factory method for SegGeometry of the segments defined by the name
   *  
   *  @param[in] segname        segment name
   *  @param[in] print_bits     print control bit-word.
   */ 

  static PSCalib::SegGeometry*
  Create ( const std::string& segname="SENS2X1:V1", const unsigned print_bits=0)
  {
	if (print_bits & 1) MsgLog("SegGeometryStore", info, "Segment geometry factory for " << segname);

        if ( segname=="SENS2X1:V1" ) { return new PSCalib::SegGeometryCspad2x1V1(); }

        //if ( segname=="SENS2X1:V2" ) { return new PSCalib::SegGeometryCspad2x1V2(); }

        //if ( segname=="SENS2X1:V3" ) { return new PSCalib::SegGeometryCspad2x1V3(); }

        if (print_bits & 2) MsgLog("SegGeometryStore", info, "Segment geometry is undefined for segment name " << segname << " - return 0-pointer...");  
        //abort();
	return 0; // NULL;
  }
};

} // namespace PSCalib

#endif // PSCALIB_SEGGEOMETRYSTORE_H
