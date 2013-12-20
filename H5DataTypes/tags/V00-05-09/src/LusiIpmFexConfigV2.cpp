//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiIpmFexConfigV2.h"

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

LusiIpmFexConfigV2::LusiIpmFexConfigV2 ( const XtcType& data )
  : xscale(data.xscale())
  , yscale(data.yscale())
{
  const ndarray<const Pds::Lusi::DiodeFexConfigV2, 1>& ndiode = data.diode();
  for ( int i = 0 ; i < XtcType::NCHANNELS ; ++ i ) {
    diode[i] = LusiDiodeFexConfigV2(ndiode[i]);
  }
}

hdf5pp::Type
LusiIpmFexConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiIpmFexConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<LusiIpmFexConfigV2>() ;
  confType.insert( "diode", offsetof(LusiIpmFexConfigV2, diode), LusiDiodeFexConfigV2::native_type(), XtcType::NCHANNELS );
  confType.insert_native<float>( "xscale", offsetof(LusiIpmFexConfigV2, xscale) );
  confType.insert_native<float>( "yscale", offsetof(LusiIpmFexConfigV2, yscale) );

  return confType ;
}

void
LusiIpmFexConfigV2::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  LusiIpmFexConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
