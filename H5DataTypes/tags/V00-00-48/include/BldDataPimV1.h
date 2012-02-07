#ifndef H5DATATYPES_BLDDATAPIMV1_H
#define H5DATATYPES_BLDDATAPIMV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPimV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Type.h"
#include "pdsdata/bld/bldData.hh"
#include "H5DataTypes/PulnixTM6740ConfigV2.h"
#include "H5DataTypes/LusiPimImageConfigV1.h"
#include "H5DataTypes/CameraFrameV1.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

namespace H5DataTypes {

//
//  Helper class for Pds::BldDataPimV1 class
//
class BldDataPimV1  {
public:

  typedef Pds::BldDataPimV1 XtcType ;

  BldDataPimV1 () {}
  BldDataPimV1 ( const XtcType& xtc ) ;

  ~BldDataPimV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  PulnixTM6740ConfigV2  camConfig;
  LusiPimImageConfigV1  pimConfig;
  CameraFrameV1         frame;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAPIMV1_H
