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
struct CsPadFilterV1_Data  {

  enum { DataSize = pdscalibdata::CsPadFilterV1::DataSize };

  uint32_t mode;
  double data[DataSize];
};

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

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

  // Data members
  CsPadFilterV1_Data m_data;

  // Copy constructor and assignment are disabled by default
  CsPadFilterV1 ( const CsPadFilterV1& ) ;
  CsPadFilterV1& operator = ( const CsPadFilterV1& ) ;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CSPADFILTERV1_H
