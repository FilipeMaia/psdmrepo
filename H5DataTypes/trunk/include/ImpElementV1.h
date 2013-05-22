#ifndef H5DATATYPES_IMPELEMENTV1_H
#define H5DATATYPES_IMPELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpElementV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/imp/ElementV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {



//
// Helper type for Pds::Imp::LaneStatus
//
class ImpSample {
public:

  typedef Pds::Imp::Sample XtcType ;

  ImpSample () {}
  ImpSample ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint32_t channels[4]; 
};

//
// Helper type for Pds::Imp::LaneStatus
//
class ImpLaneStatus  {
public:

  typedef uint32_t XtcType ;

  ImpLaneStatus () {}
  ImpLaneStatus ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint8_t linkErrCount; 
  uint8_t linkDownCount; 
  uint8_t cellErrCount; 
  uint8_t rxCount; 
  uint8_t locLinked; 
  uint8_t remLinked; 
  uint16_t zeros; 
  uint8_t powersOkay; 
};

//
// Helper type for Pds::Imp::DataV1
//
class ImpElementV1  {
public:

  typedef Pds::Imp::ElementV1 XtcType ;

  ImpElementV1 () {}
  ImpElementV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_data_type(uint32_t nSamples) ;

private:

  uint32_t vc; 
  uint32_t lane; 
  uint32_t frameNumber; 
  uint32_t range; 
  ImpLaneStatus laneStatus; 

};

} // namespace H5DataTypes

#endif // H5DATATYPES_IMPELEMENTV1_H
