//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiIpmFexConfigV1.h"

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

LusiIpmFexConfigV1::LusiIpmFexConfigV1 ( const XtcType& data )
  : xscale(data.xscale)
  , yscale(data.yscale)
{
  for ( int i = 0 ; i < XtcType::NCHANNELS ; ++ i ) {
    diode[i] = LusiDiodeFexConfigV1(data.diode[i]);
  }
}

hdf5pp::Type
LusiIpmFexConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiIpmFexConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<LusiIpmFexConfigV1>() ;
  confType.insert( "diode", offsetof(LusiIpmFexConfigV1, diode), LusiDiodeFexConfigV1::native_type(), XtcType::NCHANNELS );
  confType.insert_native<float>( "xscale", offsetof(LusiIpmFexConfigV1, xscale) );
  confType.insert_native<float>( "yscale", offsetof(LusiIpmFexConfigV1, yscale) );

  return confType ;
}

void
LusiIpmFexConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  LusiIpmFexConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
