#ifndef H5DATATYPES_CSPADMINIPIXELSTATUSV1_H
#define H5DATATYPES_CSPADMINIPIXELSTATUSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadMiniPixelStatusV1.
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
#include "pdscalibdata/CsPadMiniPixelStatusV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for pdscalibdata::CsPadMiniPixelStatusV1
//
class CsPadMiniPixelStatusV1  {
public:

  typedef pdscalibdata::CsPadMiniPixelStatusV1 DataType ;

  // Default constructor
  CsPadMiniPixelStatusV1 () ;

  // Construct from transient object
  CsPadMiniPixelStatusV1 (const DataType& data) ;

  // Destructor
  ~CsPadMiniPixelStatusV1 () ;

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

#endif // H5DATATYPES_CSPADMINIPIXELSTATUSV1_H
