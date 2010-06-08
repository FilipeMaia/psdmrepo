//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraTwoDGaussianV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CameraTwoDGaussianV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/PListDataSetCreate.h"
#include "hdf5pp/TypeTraits.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CameraTwoDGaussianV1::CameraTwoDGaussianV1 ( const Pds::Camera::TwoDGaussianV1& config )
{
  m_data.integral = config.integral() ;
  m_data.xmean = config.xmean() ;
  m_data.ymean = config.ymean() ;
  m_data.major_axis_width = config.major_axis_width() ;
  m_data.minor_axis_width = config.minor_axis_width() ;
  m_data.major_axis_tilt = config.major_axis_tilt() ;
}

hdf5pp::Type
CameraTwoDGaussianV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CameraTwoDGaussianV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CameraTwoDGaussianV1>() ;
  type.insert_native<uint64_t>( "integral", offsetof(CameraTwoDGaussianV1_Data,integral) ) ;
  type.insert_native<double>( "xmean", offsetof(CameraTwoDGaussianV1_Data,xmean) ) ;
  type.insert_native<double>( "ymean", offsetof(CameraTwoDGaussianV1_Data,ymean) ) ;
  type.insert_native<double>( "major_axis_width", offsetof(CameraTwoDGaussianV1_Data,major_axis_width) ) ;
  type.insert_native<double>( "minor_axis_width", offsetof(CameraTwoDGaussianV1_Data,minor_axis_width) ) ;
  type.insert_native<double>( "major_axis_tilt", offsetof(CameraTwoDGaussianV1_Data,major_axis_tilt) ) ;

  return type ;
}

} // namespace H5DataTypes
