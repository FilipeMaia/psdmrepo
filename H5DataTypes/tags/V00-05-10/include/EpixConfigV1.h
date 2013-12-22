#ifndef H5DATATYPES_EPIXCONFIGV1_H
#define H5DATATYPES_EPIXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixConfigV1.
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
#include "pdsdata/psddl/epix.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Epix::ConfigV1
//
class EpixConfigV1  {
public:

  enum { SchemaVersion = 0 };

  typedef Pds::Epix::ConfigV1 XtcType ;

  EpixConfigV1 () {}
  EpixConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof(); }

private:

  uint32_t version;
  uint32_t runTrigDelay;
  uint32_t daqTrigDelay;
  uint32_t dacSetting;
  uint8_t asicGR;
  uint8_t asicAcq;
  uint8_t asicR0;
  uint8_t asicPpmat;
  uint8_t asicPpbe;
  uint8_t asicRoClk;
  uint8_t asicGRControl;
  uint8_t asicAcqControl;
  uint8_t asicR0Control;
  uint8_t asicPpmatControl;
  uint8_t asicPpbeControl;
  uint8_t asicR0ClkControl;
  uint8_t prepulseR0En;
  uint32_t adcStreamMode;
  uint8_t testPatternEnable;
  uint32_t acqToAsicR0Delay;
  uint32_t asicR0ToAsicAcq;
  uint32_t asicAcqWidth;
  uint32_t asicAcqLToPPmatL;
  uint32_t asicRoClkHalfT;
  uint32_t adcReadsPerPixel;
  uint32_t adcClkHalfT;
  uint32_t asicR0Width;
  uint32_t adcPipelineDelay;
  uint32_t prepulseR0Width;
  uint32_t prepulseR0Delay;
  uint32_t digitalCardId0;
  uint32_t digitalCardId1;
  uint32_t analogCardId0;
  uint32_t analogCardId1;
  uint32_t lastRowExclusions;
  uint32_t numberOfAsicsPerRow;
  uint32_t numberOfAsicsPerColumn;
  uint32_t numberOfRowsPerAsic;
  uint32_t numberOfPixelsPerAsicRow;
  uint32_t baseClockFrequency;
  uint32_t asicMask;
  uint32_t numberOfRows;
  uint32_t numberOfColumns;
  uint32_t numberOfAsics;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPIXCONFIGV1_H
