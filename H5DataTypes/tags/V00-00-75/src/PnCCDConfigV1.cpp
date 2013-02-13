//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PnCCDConfigV1.h"

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
PnCCDConfigV1::PnCCDConfigV1 ( const PnCCDConfigV1::XtcType& config )
{
  m_data.numLinks = config.numLinks() ;
  m_data.payloadSizePerLink = config.payloadSizePerLink() ;
}



hdf5pp::Type
PnCCDConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PnCCDConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PnCCDConfigV1>() ;
  confType.insert_native<uint32_t>( "numLinks", offsetof(PnCCDConfigV1_Data,numLinks) ) ;
  confType.insert_native<uint32_t>( "payloadSizePerLink", offsetof(PnCCDConfigV1_Data,payloadSizePerLink) ) ;

  return confType ;
}

void
PnCCDConfigV1::store ( const PnCCDConfigV1::XtcType& config, hdf5pp::Group location )
{
  // make scalar data set for main object
  PnCCDConfigV1 data ( config ) ;
  storeDataObject ( data, "config", location ) ;
}


} // namespace H5DataTypes
