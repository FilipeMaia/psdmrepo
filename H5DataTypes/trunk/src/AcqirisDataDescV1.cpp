//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AcqirisDataDescV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/TypeTraits.h"

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
AcqirisDataDescV1::AcqirisDataDescV1 ()
{
}

AcqirisDataDescV1::AcqirisDataDescV1 (const AcqirisDataDescV1::XtcType& xtcData)
{
}

hdf5pp::Type
AcqirisDataDescV1::timestampType( const Pds::Acqiris::ConfigV1& config )
{
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;

  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint64_t>::native_type() ;

  hsize_t dims[] = { config.nbrChannels(), hconfig.nbrSegments() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
AcqirisDataDescV1::waveformType( const Pds::Acqiris::ConfigV1& config )
{
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;

  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { config.nbrChannels(), hconfig.nbrSegments(), hconfig.nbrSamples() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

} // namespace H5DataTypes
