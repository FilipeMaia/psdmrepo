#ifndef H5DATATYPES_ACQIRISCONFIGV1_H
#define H5DATATYPES_ACQIRISCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisConfigV1.
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
#include "pdsdata/psddl/acqiris.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Acqiris::VertV1 class
//
struct AcqirisVertV1 {
public:
  AcqirisVertV1 () {}
  AcqirisVertV1 ( const Pds::Acqiris::VertV1& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  double   fullScale;
  double   offset;
  uint32_t coupling;
  uint32_t bandwidth;
};

//
//  Helper class for Pds::Acqiris::HorizV1 class
//
struct AcqirisHorizV1 {
public:
  AcqirisHorizV1 () {}
  AcqirisHorizV1 ( const Pds::Acqiris::HorizV1& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  double   sampInterval;
  double   delayTime;
  uint32_t nbrSamples;
  uint32_t nbrSegments;
};

//
//  Helper class for Pds::Acqiris::TrigV1 class
//
struct AcqirisTrigV1 {
public:
  AcqirisTrigV1 () {}
  AcqirisTrigV1 ( const Pds::Acqiris::TrigV1& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint32_t coupling;
  uint32_t input;
  uint32_t slope;
  double   level;
};

//
//  Helper classes for Pds::Acqiris::ConfigV1 class
//
struct AcqirisConfigV1 {
public:

  typedef Pds::Acqiris::ConfigV1 XtcType ;

  AcqirisConfigV1 () {}
  AcqirisConfigV1 ( const Pds::Acqiris::ConfigV1& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const Pds::Acqiris::ConfigV1& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  uint32_t nbrConvertersPerChannel;
  uint32_t channelMask;
  uint32_t nbrChannels;
  uint32_t nbrBanks;
};


} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISCONFIGV1_H
