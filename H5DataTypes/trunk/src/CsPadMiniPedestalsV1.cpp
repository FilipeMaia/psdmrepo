//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniPedestalsV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadMiniPedestalsV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
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
CsPadMiniPedestalsV1::CsPadMiniPedestalsV1 ()
{
  // fill all pedestals with zeros
  std::fill_n(&pedestals[0][0][0], int(DataType::Size), 0.0f);
}

CsPadMiniPedestalsV1::CsPadMiniPedestalsV1 (const DataType& data)
{
  const DataType::Pedestals& pdata = data.pedestals();
  const DataType::pedestal_t* src = &pdata[0][0][0];
  DataType::pedestal_t* dst = &pedestals[0][0][0];
  std::copy(src, src+int(DataType::Size), dst );
}

//--------------
// Destructor --
//--------------
CsPadMiniPedestalsV1::~CsPadMiniPedestalsV1 ()
{
}


hdf5pp::Type
CsPadMiniPedestalsV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadMiniPedestalsV1::native_type()
{
  hsize_t dims[4] = { DataType::Columns,
                      DataType::Rows,
                      DataType::Sections};
  hdf5pp::ArrayType arrType =
    hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<DataType::pedestal_t>::native_type(), 3, dims) ;
  return arrType;
}

void
CsPadMiniPedestalsV1::store( const DataType& data, hdf5pp::Group grp, const std::string& fileName )
{
  CsPadMiniPedestalsV1 obj(data);
  hdf5pp::DataSet<CsPadMiniPedestalsV1> ds = storeDataObject ( obj, "pedestals", grp ) ;

  // add attributes
  ds.createAttr<const char*>("source").store(fileName.c_str());
}

} // namespace H5DataTypes
