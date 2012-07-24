//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadElementV1.h"

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


CsPadElementV1::CsPadElementV1 ( const XtcType& data )
  : m_data(data)
{
}

hdf5pp::Type
CsPadElementV1::stored_type(unsigned nQuad)
{
  return native_type(nQuad) ;
}

hdf5pp::Type
CsPadElementV1::native_type(unsigned nQuad)
{
  return hdf5pp::ArrayType::arrayType ( CsPadElementHeader::native_type(), nQuad );
}

hdf5pp::Type
CsPadElementV1::stored_data_type(unsigned nQuad, unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { nQuad, nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2 } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 4, dims );
}

hdf5pp::Type
CsPadElementV1::cmode_data_type(unsigned nQuad, unsigned nSect)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<float>::native_type() ;

  hsize_t dims[] = { nQuad, nSect } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
