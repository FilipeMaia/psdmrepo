#ifndef H5DATATYPES_ORCACONFIGV1_H
#define H5DATATYPES_ORCACONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OrcaConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/orca/ConfigV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Orca::ConfigV1
//
class OrcaConfigV1  {
public:

  typedef Pds::Orca::ConfigV1 XtcType ;

  OrcaConfigV1 () {}
  OrcaConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t rows;
  uint8_t mode;
  uint8_t cooling;
  uint8_t defect_pixel_correction_enabled;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_ORCACONFIGV1_H
