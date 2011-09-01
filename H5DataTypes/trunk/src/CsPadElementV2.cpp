//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadElementV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"
#include "pdsdata/cspad/Detector.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CsPadElementV2::CsPadElementV2 ( const XtcType& data )
  : m_data(data)
{
}

hdf5pp::Type
CsPadElementV2::stored_type(unsigned nQuad)
{
  return native_type(nQuad) ;
}

hdf5pp::Type
CsPadElementV2::native_type(unsigned nQuad)
{
  return hdf5pp::ArrayType::arrayType ( CsPadElementHeader::native_type(), nQuad );
}

hdf5pp::Type
CsPadElementV2::stored_data_type(unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

hdf5pp::Type
CsPadElementV2::cmode_data_type(unsigned nSect)
{
  return hdf5pp::ArrayType::arrayType<float> ( nSect );
}

} // namespace H5DataTypes
