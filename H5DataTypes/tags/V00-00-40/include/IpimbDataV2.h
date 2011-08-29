#ifndef H5DATATYPES_IPIMBDATAV2_H
#define H5DATATYPES_IPIMBDATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbDataV2.
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
#include "pdsdata/ipimb/DataV2.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Ipimb::DataV2
//
struct IpimbDataV2_Data  {
  uint64_t triggerCounter;
  uint16_t config0;
  uint16_t config1;
  uint16_t config2;
  uint16_t channel0;
  uint16_t channel1;
  uint16_t channel2;
  uint16_t channel3;
  uint16_t channel0ps;
  uint16_t channel1ps;
  uint16_t channel2ps;
  uint16_t channel3ps;
  uint16_t checksum;
  float channel0Volts;
  float channel1Volts;
  float channel2Volts;
  float channel3Volts;
  float channel0psVolts;
  float channel1psVolts;
  float channel2psVolts;
  float channel3psVolts;
};

class IpimbDataV2  {
public:

  typedef Pds::Ipimb::DataV2 XtcType ;

  IpimbDataV2 () {}
  IpimbDataV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  IpimbDataV2_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_IPIMBDATAV2_H
