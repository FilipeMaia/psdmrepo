#ifndef H5DATATYPES_ACQIRISDATADESCV1_H
#define H5DATATYPES_ACQIRISDATADESCV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/acqiris/ConfigV1.hh"
#include "pdsdata/acqiris/DataDescV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Acqiris::TimestampV1 class
//
class AcqirisTimestampV1  {
public:

  typedef Pds::Acqiris::TimestampV1 XtcType ;

  AcqirisTimestampV1 () : value(0), pos(0) {}
  AcqirisTimestampV1 ( const XtcType& xtcData ) ;
  
  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

protected:
private:
  
  uint64_t value;
  double pos;

};

//
//  Helper class for Pds::Acqiris::DataDescV1 class
//
class AcqirisDataDescV1  {
public:

  typedef Pds::Acqiris::DataDescV1 XtcType ;

  AcqirisDataDescV1 () : nbrSamplesInSeg(0), nbrSegments(0), indexFirstPoint(0) {}
  AcqirisDataDescV1 ( const XtcType& xtcData ) ;
  
  static hdf5pp::Type stored_type(const Pds::Acqiris::ConfigV1& config) ;
  static hdf5pp::Type native_type(const Pds::Acqiris::ConfigV1& config) ;

  static hdf5pp::Type timestampType( const Pds::Acqiris::ConfigV1& config ) ;
  static hdf5pp::Type waveformType( const Pds::Acqiris::ConfigV1& config ) ;

protected:
private:
  
  uint32_t nbrSamplesInSeg;
  uint32_t nbrSegments;
  uint32_t indexFirstPoint;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISDATADESCV1_H
