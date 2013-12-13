#ifndef H5DATATYPES_EPIXSAMPLERCONFIGV1_H
#define H5DATATYPES_EPIXSAMPLERCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerConfigV1.
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
#include "pdsdata/psddl/epixsampler.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EpixSampler::ConfigV1
//
class EpixSamplerConfigV1  {
public:

  typedef Pds::EpixSampler::ConfigV1 XtcType ;

  EpixSamplerConfigV1 () {}
  EpixSamplerConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t version;
  uint32_t runTrigDelay;
  uint32_t daqTrigDelay;
  uint32_t daqSetting;
  uint32_t adcClkHalfT;
  uint32_t adcPipelineDelay;
  uint32_t digitalCardId0;
  uint32_t digitalCardId1;
  uint32_t analogCardId0;
  uint32_t analogCardId1;
  uint32_t numberOfChannels;
  uint32_t samplesPerChannel;
  uint32_t baseClockFrequency;
  uint8_t testPatternEnable;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPIXSAMPLERCONFIGV1_H
