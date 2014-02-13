#ifndef H5DATATYPES_EPIXSAMPLERELEMENTV1_H
#define H5DATATYPES_EPIXSAMPLERELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerElementV1.
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
#include "hdf5pp/Type.h"
#include "pdsdata/psddl/epixsampler.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EpixSampler::ElementV1
//
class EpixSamplerElementV1  {
public:

  enum { SchemaVersion = 0 };

  typedef Pds::EpixSampler::ElementV1 XtcType ;

  EpixSamplerElementV1 () {}
  EpixSamplerElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type frame_data_type(int nChan, int samplesPerChannel) ;
  static hdf5pp::Type temperature_data_type(int nChan) ;

private:

  uint8_t vc;
  uint8_t lane;
  uint16_t acqCount;
  uint32_t frameNumber;
  uint32_t ticks;
  uint32_t fiducials;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPIXSAMPLERELEMENTV1_H
