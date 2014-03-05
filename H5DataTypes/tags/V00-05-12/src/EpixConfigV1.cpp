//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpixConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "hdf5pp/Utils.h"
#include "H5DataTypes/EpixAsicConfigV1.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

EpixConfigV1::EpixConfigV1 ( const Pds::Epix::ConfigV1& data )
  : version(data.version())
  , runTrigDelay(data.runTrigDelay())
  , daqTrigDelay(data.daqTrigDelay())
  , dacSetting(data.dacSetting())
  , asicGR(data.asicGR())
  , asicAcq(data.asicAcq())
  , asicR0(data.asicR0())
  , asicPpmat(data.asicPpmat())
  , asicPpbe(data.asicPpbe())
  , asicRoClk(data.asicRoClk())
  , asicGRControl(data.asicGRControl())
  , asicAcqControl(data.asicAcqControl())
  , asicR0Control(data.asicR0Control())
  , asicPpmatControl(data.asicPpmatControl())
  , asicPpbeControl(data.asicPpbeControl())
  , asicR0ClkControl(data.asicR0ClkControl())
  , prepulseR0En(data.prepulseR0En())
  , adcStreamMode(data.adcStreamMode())
  , testPatternEnable(data.testPatternEnable())
  , acqToAsicR0Delay(data.acqToAsicR0Delay())
  , asicR0ToAsicAcq(data.asicR0ToAsicAcq())
  , asicAcqWidth(data.asicAcqWidth())
  , asicAcqLToPPmatL(data.asicAcqLToPPmatL())
  , asicRoClkHalfT(data.asicRoClkHalfT())
  , adcReadsPerPixel(data.adcReadsPerPixel())
  , adcClkHalfT(data.adcClkHalfT())
  , asicR0Width(data.asicR0Width())
  , adcPipelineDelay(data.adcPipelineDelay())
  , prepulseR0Width(data.prepulseR0Width())
  , prepulseR0Delay(data.prepulseR0Delay())
  , digitalCardId0(data.digitalCardId0())
  , digitalCardId1(data.digitalCardId1())
  , analogCardId0(data.analogCardId0())
  , analogCardId1(data.analogCardId1())
  , lastRowExclusions(data.lastRowExclusions())
  , numberOfAsicsPerRow(data.numberOfAsicsPerRow())
  , numberOfAsicsPerColumn(data.numberOfAsicsPerColumn())
  , numberOfRowsPerAsic(data.numberOfRowsPerAsic())
  , numberOfPixelsPerAsicRow(data.numberOfPixelsPerAsicRow())
  , baseClockFrequency(data.baseClockFrequency())
  , asicMask(data.asicMask())
  , numberOfRows(data.numberOfRows())
  , numberOfColumns(data.numberOfColumns())
  , numberOfAsics(data.numberOfAsics())
{
}

hdf5pp::Type
EpixConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpixConfigV1::native_type()
{
  typedef EpixConfigV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("version", offsetof(DsType, version), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("runTrigDelay", offsetof(DsType, runTrigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("daqTrigDelay", offsetof(DsType, daqTrigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("dacSetting", offsetof(DsType, dacSetting), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicGR", offsetof(DsType, asicGR), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicAcq", offsetof(DsType, asicAcq), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicR0", offsetof(DsType, asicR0), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicPpmat", offsetof(DsType, asicPpmat), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicPpbe", offsetof(DsType, asicPpbe), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicRoClk", offsetof(DsType, asicRoClk), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicGRControl", offsetof(DsType, asicGRControl), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicAcqControl", offsetof(DsType, asicAcqControl), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicR0Control", offsetof(DsType, asicR0Control), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicPpmatControl", offsetof(DsType, asicPpmatControl), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicPpbeControl", offsetof(DsType, asicPpbeControl), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("asicR0ClkControl", offsetof(DsType, asicR0ClkControl), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("prepulseR0En", offsetof(DsType, prepulseR0En), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("adcStreamMode", offsetof(DsType, adcStreamMode), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("testPatternEnable", offsetof(DsType, testPatternEnable), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("acqToAsicR0Delay", offsetof(DsType, acqToAsicR0Delay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicR0ToAsicAcq", offsetof(DsType, asicR0ToAsicAcq), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicAcqWidth", offsetof(DsType, asicAcqWidth), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicAcqLToPPmatL", offsetof(DsType, asicAcqLToPPmatL), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicRoClkHalfT", offsetof(DsType, asicRoClkHalfT), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcReadsPerPixel", offsetof(DsType, adcReadsPerPixel), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcClkHalfT", offsetof(DsType, adcClkHalfT), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicR0Width", offsetof(DsType, asicR0Width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcPipelineDelay", offsetof(DsType, adcPipelineDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("prepulseR0Width", offsetof(DsType, prepulseR0Width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("prepulseR0Delay", offsetof(DsType, prepulseR0Delay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("digitalCardId0", offsetof(DsType, digitalCardId0), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("digitalCardId1", offsetof(DsType, digitalCardId1), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("analogCardId0", offsetof(DsType, analogCardId0), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("analogCardId1", offsetof(DsType, analogCardId1), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("lastRowExclusions", offsetof(DsType, lastRowExclusions), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfAsicsPerRow", offsetof(DsType, numberOfAsicsPerRow), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfAsicsPerColumn", offsetof(DsType, numberOfAsicsPerColumn), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfRowsPerAsic", offsetof(DsType, numberOfRowsPerAsic), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfPixelsPerAsicRow", offsetof(DsType, numberOfPixelsPerAsicRow), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("baseClockFrequency", offsetof(DsType, baseClockFrequency), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("asicMask", offsetof(DsType, asicMask), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfRows", offsetof(DsType, numberOfRows), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfColumns", offsetof(DsType, numberOfColumns), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numberOfAsics", offsetof(DsType, numberOfAsics), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

void
EpixConfigV1::store( const Pds::Epix::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EpixConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // asics config objects
  const int nasics = config.numberOfAsics();
  EpixAsicConfigV1 asics[nasics];
  for (int i = 0; i != nasics; ++ i) {
    asics[i] = EpixAsicConfigV1(config.asics(i));
  }
  storeDataObjects(nasics, asics, "asics", grp);

  hdf5pp::Utils::storeNDArray(grp, "asicPixelTestArray", config.asicPixelTestArray());

  hdf5pp::Utils::storeNDArray(grp, "asicPixelMaskArray", config.asicPixelMaskArray());
}

} // namespace H5DataTypes
