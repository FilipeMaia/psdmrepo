#ifndef H5DATATYPES_IPIMBDATAV1_H
#define H5DATATYPES_IPIMBDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbDataV1.
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
// Helper type for Pds::Ipimb::DataV1
//
class IpimbDataV1  {
public:

  typedef Pds::Ipimb::DataV1 XtcType ;

  IpimbDataV1 () {}
  IpimbDataV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint64_t triggerCounter;
  uint16_t config0;
  uint16_t config1;
  uint16_t config2;
  uint16_t channel0;
  uint16_t channel1;
  uint16_t channel2;
  uint16_t channel3;
  uint16_t checksum;
  float channel0Volts;
  float channel1Volts;
  float channel2Volts;
  float channel3Volts;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_IPIMBDATAV1_H
