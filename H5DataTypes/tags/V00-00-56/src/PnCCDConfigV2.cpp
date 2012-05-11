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
{
  m_data.numLinks = config.numLinks();
  m_data.payloadSizePerLink = config.payloadSizePerLink();
  m_data.numChannels = config.numChannels();
  m_data.numRows = config.numRows();
  m_data.numSubmoduleChannels = config.numSubmoduleChannels();
  m_data.numSubmoduleRows = config.numSubmoduleRows();
  m_data.numSubmodules = config.numSubmodules();
  m_data.camexMagic = config.camexMagic();

  const char* p = config.info();
  int len = strlen(p)+1;
  m_data.info = new char[len];
  std::copy(p, p+len, m_data.info);

  p = config.timingFName();
  len = strlen(p)+1;
  m_data.timingFName = new char[len];
  std::copy( p, p+len, m_data.timingFName);
}

PnCCDConfigV2::~PnCCDConfigV2()
{
  delete [] m_data.info;
  delete [] m_data.timingFName;
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
  confType.insert_native<uint32_t>( "numLinks", offsetof(PnCCDConfigV2_Data,numLinks) ) ;
  confType.insert_native<uint32_t>( "payloadSizePerLink", offsetof(PnCCDConfigV2_Data,payloadSizePerLink) ) ;
  confType.insert_native<uint32_t>( "numChannels", offsetof(PnCCDConfigV2_Data,numChannels) ) ;
  confType.insert_native<uint32_t>( "numRows", offsetof(PnCCDConfigV2_Data,numRows) ) ;
  confType.insert_native<uint32_t>( "numSubmoduleChannels", offsetof(PnCCDConfigV2_Data,numSubmoduleChannels) ) ;
  confType.insert_native<uint32_t>( "numSubmoduleRows", offsetof(PnCCDConfigV2_Data,numSubmoduleRows) ) ;
  confType.insert_native<uint32_t>( "numSubmodules", offsetof(PnCCDConfigV2_Data,numSubmodules) ) ;
  confType.insert_native<uint32_t>( "camexMagic", offsetof(PnCCDConfigV2_Data,camexMagic) ) ;
  confType.insert_native<const char*>( "info", offsetof(PnCCDConfigV2_Data,info) ) ;
  confType.insert_native<const char*>( "timingFName", offsetof(PnCCDConfigV2_Data,timingFName) ) ;

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
