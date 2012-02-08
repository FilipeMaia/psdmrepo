#ifndef H5DATATYPES_CSPAD2X2PEDESTALSV1_H
#define H5DATATYPES_CSPAD2X2PEDESTALSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PedestalsV1.
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
#include "pdscalibdata/CsPad2x2PedestalsV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for pdscalibdata::CsPad2x2PedestalsV1
//
class CsPad2x2PedestalsV1  {
public:

  typedef pdscalibdata::CsPad2x2PedestalsV1 DataType ;

  // Default constructor
  CsPad2x2PedestalsV1 () ;

  // Construct from transient object
  CsPad2x2PedestalsV1 (const DataType& data) ;

  // Destructor
  ~CsPad2x2PedestalsV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single object at specified location
  static void store( const DataType& data,
                     hdf5pp::Group location,
                     const std::string& fileName = std::string()) ;
  
protected:

private:

  DataType::pedestal_t pedestals[DataType::Columns][DataType::Rows][DataType::Sections];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CSPAD2X2PEDESTALSV1_H
