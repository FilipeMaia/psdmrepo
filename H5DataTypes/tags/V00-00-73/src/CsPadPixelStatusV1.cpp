//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelStatusV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPadPixelStatusV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ErrSvc/Issue.h"
#include "hdf5pp/ArrayType.h"
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
CsPadPixelStatusV1::CsPadPixelStatusV1 ()
{
  // fill all codes with zeros
  pdscalibdata::CsPadPixelStatusV1::status_t zero=0;
  std::fill_n(&status[0][0][0][0], int(DataType::Size), zero);
}

CsPadPixelStatusV1::CsPadPixelStatusV1 (const DataType& data) 
{ 
  const ndarray<DataType::status_t, 4>& sdata = data.status();

  // verify that data shape is what we expect
  const unsigned* shape = sdata.shape();
  if (shape[0] != DataType::Quads or shape[1] != DataType::Sections or
      shape[2] != DataType::Columns or shape[3] != DataType::Rows) {
    throw BadShape(ERR_LOC);
  }

  const DataType::status_t* src = &sdata[0][0][0][0];
  DataType::status_t* dst = &status[0][0][0][0];
  std::copy(src, src+int(DataType::Size), dst );
}

//--------------
// Destructor --
//--------------
CsPadPixelStatusV1::~CsPadPixelStatusV1 ()
{
}


hdf5pp::Type
CsPadPixelStatusV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPadPixelStatusV1::native_type()
{
  hsize_t dims[4] = { DataType::Quads,
                      DataType::Sections,
                      DataType::Columns,
                      DataType::Rows};
  hdf5pp::ArrayType arrType = 
    hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<DataType::status_t>::native_type(), 4, dims) ;
  return arrType;
}

void
CsPadPixelStatusV1::store( const DataType& data, hdf5pp::Group grp, const std::string& fileName )
{
  CsPadPixelStatusV1 obj(data);
  hdf5pp::DataSet<CsPadPixelStatusV1> ds = storeDataObject ( obj, "pixel_status", grp ) ;
  
  // add attributes
  ds.createAttr<const char*>("source").store(fileName.c_str());
}

} // namespace H5DataTypes
