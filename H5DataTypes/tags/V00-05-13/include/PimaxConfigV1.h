#ifndef H5DATATYPES_PIMAXCONFIGV1_H
#define H5DATATYPES_PIMAXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PimaxConfigV1.
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
#include "pdsdata/psddl/pimax.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Pimax::ConfigV1
//
class PimaxConfigV1  {
public:

  typedef Pds::Pimax::ConfigV1 XtcType ;

  PimaxConfigV1 () {}
  PimaxConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t width;
  uint32_t height;
  uint32_t orgX;
  uint32_t orgY;
  uint32_t binX;
  uint32_t binY;
  float exposureTime;
  float coolingTemp;
  float readoutSpeed;
  uint16_t gainIndex;
  uint16_t intesifierGain;
  double gateDelay;
  double gateWidth;
  uint32_t maskedHeight;
  uint32_t kineticHeight;
  float vsSpeed;
  int16_t infoReportInterval;
  uint16_t exposureEventCode;
  uint32_t numIntegrationShots;
  uint32_t frameSize;
  uint32_t numPixelsX;
  uint32_t numPixelsY;
  uint32_t numPixels;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PIMAXCONFIGV1_H
