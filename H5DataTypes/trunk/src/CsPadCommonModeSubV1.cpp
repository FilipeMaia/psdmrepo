//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCommonModeSubV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadCommonModeSubV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"
#include "MsgLogger/MsgLogger.h"

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
CsPadCommonModeSubV1::CsPadCommonModeSubV1 ()
  : mode(0)
{
  // fill all codes with zeros
  std::fill_n(data, int(DataSize), 0.0);
}

CsPadCommonModeSubV1::CsPadCommonModeSubV1 (const DataType& data) 
  : mode(data.mode())
{ 
  const double* src = data.data();
  std::copy(src, src+int(DataSize), this->data );
}

//--------------
// Destructor --
//--------------
CsPadCommonModeSubV1::~CsPadCommonModeSubV1 ()
{
}


hdf5pp::Type
CsPadCommonModeSubV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadCommonModeSubV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CsPadCommonModeSubV1>() ;
  type.insert_native<uint32_t>( "mode", offsetof(CsPadCommonModeSubV1,mode) ) ;
  type.insert_native<double>( "data", offsetof(CsPadCommonModeSubV1,data), DataSize ) ;

  return type;
}

void
CsPadCommonModeSubV1::store( const DataType& data, hdf5pp::Group grp, const std::string& fileName )
{
  CsPadCommonModeSubV1 obj(data);
  hdf5pp::DataSet ds = storeDataObject ( obj, "common_mode", grp ) ;
  
  // add attributes
  ds.createAttr<const char*>("source").store(fileName.c_str());
}

} // namespace H5DataTypes
