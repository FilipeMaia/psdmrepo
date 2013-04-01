#ifndef H5DATATYPES_CSPADFILTERV1_H
#define H5DATATYPES_CSPADFILTERV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadFilterV1.
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
#include "pdscalibdata/CsPadFilterV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for CsPadFilterV1
//
class CsPadFilterV1  {
public:

  typedef pdscalibdata::CsPadFilterV1 DataType ;
  
  // Default constructor
  CsPadFilterV1 () ;
  
  // Construct from transient object
  CsPadFilterV1 (const DataType& data) ;

  // Destructor
  ~CsPadFilterV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single object at specified location
  static void store( const DataType& data, 
                     hdf5pp::Group location,
                     const std::string& fileName = std::string()) ;
  
protected:

private:

  enum { DataSize = pdscalibdata::CsPadFilterV1::DataSize };

  uint32_t mode;
  double data[DataSize];

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADFILTERV1_H
