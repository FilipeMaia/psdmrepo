//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelStatusV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CsPad2x2PixelStatusV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>

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
CsPad2x2PixelStatusV1::CsPad2x2PixelStatusV1 ()
{
  // fill all codes with zeros
  pdscalibdata::CsPad2x2PixelStatusV1::status_t zero=0;
  std::fill_n(&status[0][0][0], int(DataType::Size), zero);
}

CsPad2x2PixelStatusV1::CsPad2x2PixelStatusV1 (const DataType& data)
{
  const ndarray<DataType::status_t, 3>& sdata = data.status();

  // verify that data shape is what we expect
  const unsigned* shape = sdata.shape();
  if (shape[0] != DataType::Columns or shape[1] != DataType::Rows or shape[2] != DataType::Sections) {
    throw BadShape(ERR_LOC);
  }

  const DataType::status_t* src = sdata.data();
  DataType::status_t* dst = &status[0][0][0];
  std::copy(src, src+int(DataType::Size), dst );
}

//--------------
// Destructor --
//--------------
CsPad2x2PixelStatusV1::~CsPad2x2PixelStatusV1 ()
{
}


hdf5pp::Type
CsPad2x2PixelStatusV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CsPad2x2PixelStatusV1::native_type()
{
  hsize_t dims[4] = { DataType::Columns,
                      DataType::Rows,
                      DataType::Sections};
  hdf5pp::ArrayType arrType =
    hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<DataType::status_t>::native_type(), 3, dims) ;
  return arrType;
}

void
CsPad2x2PixelStatusV1::store( const DataType& data, hdf5pp::Group grp, const std::string& fileName )
{
  CsPad2x2PixelStatusV1 obj(data);
  hdf5pp::DataSet<CsPad2x2PixelStatusV1> ds = storeDataObject ( obj, "pixel_status", grp ) ;

  // add attributes
  ds.createAttr<const char*>("source").store(fileName.c_str());
}

} // namespace H5DataTypes
