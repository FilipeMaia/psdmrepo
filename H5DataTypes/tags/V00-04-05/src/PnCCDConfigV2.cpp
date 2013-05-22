//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PnCCDConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
PnCCDConfigV2::PnCCDConfigV2 ( const PnCCDConfigV2::XtcType& config )
  : numLinks(config.numLinks())
  , payloadSizePerLink(config.payloadSizePerLink())
  , numChannels(config.numChannels())
  , numRows(config.numRows())
  , numSubmoduleChannels(config.numSubmoduleChannels())
  , numSubmoduleRows(config.numSubmoduleRows())
  , numSubmodules(config.numSubmodules())
  , camexMagic(config.camexMagic())
{
  const char* p = config.info();
  int len = strlen(p)+1;
  info = new char[len];
  std::copy(p, p+len, info);

  p = config.timingFName();
  len = strlen(p)+1;
  timingFName = new char[len];
  std::copy(p, p+len, timingFName);
}

PnCCDConfigV2::~PnCCDConfigV2()
{
  delete [] info;
  delete [] timingFName;
}

hdf5pp::Type
PnCCDConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PnCCDConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PnCCDConfigV2>() ;
  confType.insert_native<uint32_t>( "numLinks", offsetof(PnCCDConfigV2,numLinks) ) ;
  confType.insert_native<uint32_t>( "payloadSizePerLink", offsetof(PnCCDConfigV2,payloadSizePerLink) ) ;
  confType.insert_native<uint32_t>( "numChannels", offsetof(PnCCDConfigV2,numChannels) ) ;
  confType.insert_native<uint32_t>( "numRows", offsetof(PnCCDConfigV2,numRows) ) ;
  confType.insert_native<uint32_t>( "numSubmoduleChannels", offsetof(PnCCDConfigV2,numSubmoduleChannels) ) ;
  confType.insert_native<uint32_t>( "numSubmoduleRows", offsetof(PnCCDConfigV2,numSubmoduleRows) ) ;
  confType.insert_native<uint32_t>( "numSubmodules", offsetof(PnCCDConfigV2,numSubmodules) ) ;
  confType.insert_native<uint32_t>( "camexMagic", offsetof(PnCCDConfigV2,camexMagic) ) ;
  confType.insert_native<const char*>( "info", offsetof(PnCCDConfigV2,info) ) ;
  confType.insert_native<const char*>( "timingFName", offsetof(PnCCDConfigV2,timingFName) ) ;

  return confType ;
}

void
PnCCDConfigV2::store ( const PnCCDConfigV2::XtcType& config, hdf5pp::Group location )
{
  // make scalar data set for main object
  PnCCDConfigV2 data ( config ) ;
  storeDataObject ( data, "config", location ) ;
}

} // namespace H5DataTypes
