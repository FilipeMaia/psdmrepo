#ifndef H5DATATYPES_CSPADPEDESTALSV1_H
#define H5DATATYPES_CSPADPEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestalsV1.
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
#include "pdscalibdata/CsPadPedestalsV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for pdscalibdata::CsPadPedestalsV1
//
class CsPadPedestalsV1  {
public:

  typedef pdscalibdata::CsPadPedestalsV1 DataType ;
  
  // Default constructor
  CsPadPedestalsV1 () ;
  
  // Construct from transient object
  CsPadPedestalsV1 (const DataType& data) ;

  // Destructor
  ~CsPadPedestalsV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single object at specified location
  static void store( const DataType& data, 
                     hdf5pp::Group location,
                     const std::string& fileName = std::string()) ;
  
protected:

private:

  DataType::Pedestals pedestals;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADPEDESTALSV1_H
