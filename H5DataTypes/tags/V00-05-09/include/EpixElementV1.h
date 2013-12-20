#ifndef H5DATATYPES_EPIXELEMENTV1_H
#define H5DATATYPES_EPIXELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixElementV1.
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
#include "pdsdata/psddl/epix.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Epix::ElementV1
//
class EpixElementV1  {
public:

  enum { SchemaVersion = 0 };

  typedef Pds::Epix::ElementV1 XtcType ;

  EpixElementV1 () {}

  template <typename ConfigType>
  EpixElementV1 ( const XtcType& data, const ConfigType& config )
    : vc(data.vc())
    , lane(data.lane())
    , acqCount(data.acqCount())
    , frameNumber(data.frameNumber())
    , ticks(data.ticks())
    , fiducials(data.fiducials())
    , lastWord(data.lastWord(config))
  {
  }

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type frame_data_type(int numberOfRows, int numberOfColumns) ;
  static hdf5pp::Type excludedRows_data_type(int lastRowExclusions, int numberOfColumns) ;
  static hdf5pp::Type temperature_data_type(int nAsics) ;

private:

  uint8_t vc;
  uint8_t lane;
  uint16_t acqCount;
  uint32_t frameNumber;
  uint32_t ticks;
  uint32_t fiducials;
  uint32_t lastWord;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPIXELEMENTV1_H
