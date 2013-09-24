#ifndef H5DATATYPES_IPIMBCONFIGV1_H
#define H5DATATYPES_IPIMBCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbConfigV1.
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
#include "pdsdata/psddl/ipimb.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Ipimb::ConfigV1
//
class IpimbConfigV1  {
public:

  typedef Pds::Ipimb::ConfigV1 XtcType ;

  IpimbConfigV1 () {}
  IpimbConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint64_t triggerCounter;
  uint64_t serialID;
  uint16_t chargeAmpRange;
  uint16_t calibrationRange;
  uint8_t capacitorValue[4];
  uint32_t resetLength;
  uint16_t resetDelay;
  float chargeAmpRefVoltage;
  float calibrationVoltage;
  float diodeBias;
  uint16_t status;
  uint16_t errors;
  uint16_t calStrobeLength;
  uint32_t trigDelay;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_IPIMBCONFIGV1_H
