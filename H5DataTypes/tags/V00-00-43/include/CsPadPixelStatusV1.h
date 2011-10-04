#ifndef H5DATATYPES_CSPADPIXELSTATUSV1_H
#define H5DATATYPES_CSPADPIXELSTATUSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelStatusV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for pdscalibdata::CsPadPixelStatusV1
//
class CsPadPixelStatusV1  {
public:

  typedef pdscalibdata::CsPadPixelStatusV1 DataType ;
  
  // Default constructor
  CsPadPixelStatusV1 () ;
  
  // Construct from transient object
  CsPadPixelStatusV1 (const DataType& data) ;

  // Destructor
  ~CsPadPixelStatusV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single object at specified location
  static void store( const DataType& data, 
                     hdf5pp::Group location,
                     const std::string& fileName = std::string()) ;
  
protected:

private:

  DataType::StatusCodes status;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADPIXELSTATUSV1_H
