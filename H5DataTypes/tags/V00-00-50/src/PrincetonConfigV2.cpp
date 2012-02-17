//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

PrincetonConfigV2::PrincetonConfigV2 ( const Pds::Princeton::ConfigV2& data )
{
  m_data.width = data.width();
  m_data.height = data.height();
  m_data.orgX = data.orgX();
  m_data.orgY = data.orgY();
  m_data.binX = data.binX();
  m_data.binY = data.binY();
  m_data.exposureTime = data.exposureTime();
  m_data.coolingTemp = data.coolingTemp();
  m_data.gainIndex = data.gainIndex();
  m_data.readoutSpeedIndex = data.readoutSpeedIndex();
  m_data.readoutEventCode = data.readoutEventCode();
  m_data.delayMode = data.delayMode();
}

hdf5pp::Type
PrincetonConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV2>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV2_Data,width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV2_Data,height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV2_Data,orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV2_Data,orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV2_Data,binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV2_Data,binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV2_Data,exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV2_Data,coolingTemp) );
  confType.insert_native<uint16_t>( "gainIndex", offsetof(PrincetonConfigV2_Data,gainIndex) );
  confType.insert_native<uint16_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV2_Data,readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "readoutEventCode", offsetof(PrincetonConfigV2_Data,readoutEventCode) );
  confType.insert_native<uint16_t>( "delayMode", offsetof(PrincetonConfigV2_Data,delayMode) );

  return confType ;
}

void
PrincetonConfigV2::store( const Pds::Princeton::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
