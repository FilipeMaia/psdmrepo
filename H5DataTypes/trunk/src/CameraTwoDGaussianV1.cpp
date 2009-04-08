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
#include "Lusi/Lusi.h"

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
#include "hdf5pp/PListDataSetCreate.h"
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
CameraTwoDGaussianV1::persType()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CameraTwoDGaussianV1>() ;
  confType.insert( "integral", offsetof(CameraTwoDGaussianV1_Data,integral), hdf5pp::AtomicType::atomicType<uint64_t>() ) ;
  confType.insert( "xmean", offsetof(CameraTwoDGaussianV1_Data,xmean), hdf5pp::AtomicType::atomicType<double>() ) ;
  confType.insert( "ymean", offsetof(CameraTwoDGaussianV1_Data,ymean), hdf5pp::AtomicType::atomicType<double>() ) ;
  confType.insert( "major_axis_width", offsetof(CameraTwoDGaussianV1_Data,major_axis_width), hdf5pp::AtomicType::atomicType<double>() ) ;
  confType.insert( "minor_axis_width", offsetof(CameraTwoDGaussianV1_Data,minor_axis_width), hdf5pp::AtomicType::atomicType<double>() ) ;
  confType.insert( "major_axis_tilt", offsetof(CameraTwoDGaussianV1_Data,major_axis_tilt), hdf5pp::AtomicType::atomicType<double>() ) ;

  return confType ;
}

} // namespace H5DataTypes
