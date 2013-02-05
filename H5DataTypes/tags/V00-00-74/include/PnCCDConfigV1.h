#ifndef H5DATATYPES_PNCCDCONFIGV1_H
#define H5DATATYPES_PNCCDCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/pnCCD/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::PNCCD::ConfigV1
//
struct PnCCDConfigV1_Data {
  uint32_t numLinks;
  uint32_t payloadSizePerLink;
};

class PnCCDConfigV1 {
public:

  typedef Pds::PNCCD::ConfigV1 XtcType ;

  PnCCDConfigV1() {}
  PnCCDConfigV1 ( const XtcType& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(XtcType) ; }

protected:

private:

  PnCCDConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PNCCDCONFIGV1_H
