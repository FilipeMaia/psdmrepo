//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestalsV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadPedestalsV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ErrSvc/Issue.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  class BadShape: public ErrSvc::Issue {
  public:
    BadShape(const ErrSvc::Context& ctx) : ErrSvc::Issue(ctx, "Illegal shape of data array") {}
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
CsPadPedestalsV1::CsPadPedestalsV1 ()
{
  // fill all pedestals with zeros
  std::fill_n(&pedestals[0][0][0][0], int(DataType::Size), 0.0f);
}

CsPadPedestalsV1::CsPadPedestalsV1 (const DataType& data) 
{ 
  const ndarray<DataType::pedestal_t, 4>& pdata = data.pedestals();

  // verify that data shape is what we expect
  const unsigned* shape = pdata.shape();
  if (shape[0] != DataType::Quads or shape[1] != DataType::Sections or
      shape[2] != DataType::Columns or shape[3] != DataType::Rows) {
    throw BadShape(ERR_LOC);
  }

  const DataType::pedestal_t* src = &pdata[0][0][0][0];
  DataType::pedestal_t* dst = &pedestals[0][0][0][0];
  std::copy(src, src+int(DataType::Size), dst );
}

//--------------
// Destructor --
//--------------
CsPadPedestalsV1::~CsPadPedestalsV1 ()
{
}


hdf5pp::Type
CsPadPedestalsV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadPedestalsV1::native_type()
{
  hsize_t dims[4] = { DataType::Quads,
                      DataType::Sections,
                      DataType::Columns,
                      DataType::Rows};
  hdf5pp::ArrayType arrType = 
    hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<DataType::pedestal_t>::native_type(), 4, dims) ;
  return arrType;
}

void
CsPadPedestalsV1::store( const DataType& data, hdf5pp::Group grp, const std::string& fileName )
{
  CsPadPedestalsV1 obj(data);
  hdf5pp::DataSet ds = storeDataObject ( obj, "pedestals", grp ) ;

  // add attributes
  ds.createAttr<const char*>("source").store(fileName.c_str());
}

} // namespace H5DataTypes
