//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV1.h"

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

PrincetonConfigV1::PrincetonConfigV1 ( const Pds::Princeton::ConfigV1& data )
{
  m_data.width = data.width();
  m_data.height = data.height();
  m_data.orgX = data.orgX();
  m_data.orgY = data.orgY();
  m_data.binX = data.binX();
  m_data.binY = data.binY();
  m_data.exposureTime = data.exposureTime();
  m_data.coolingTemp = data.coolingTemp();
  m_data.readoutSpeedIndex = data.readoutSpeedIndex();
  m_data.readoutEventCode = data.readoutEventCode();
  m_data.delayMode = data.delayMode();
}

hdf5pp::Type
PrincetonConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV1>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV1_Data,width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV1_Data,height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV1_Data,orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV1_Data,orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV1_Data,binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV1_Data,binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV1_Data,exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV1_Data,coolingTemp) );
  confType.insert_native<uint32_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV1_Data,readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "readoutEventCode", offsetof(PrincetonConfigV1_Data,readoutEventCode) );
  confType.insert_native<uint16_t>( "delayMode", offsetof(PrincetonConfigV1_Data,delayMode) );

  return confType ;
}

void
PrincetonConfigV1::store( const Pds::Princeton::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
