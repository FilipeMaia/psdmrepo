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
#include "pdsdata/acqiris/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

//
//  Helper classes for Pds::Acqiris::VertV1 class
//
struct AcqirisVertV1_Data {
  double   fullScale;
  double   offset;
  uint32_t coupling;
  uint32_t bandwidth;
};

struct AcqirisVertV1 {
public:
  AcqirisVertV1 () {}
  AcqirisVertV1 ( const Pds::Acqiris::VertV1& v ) ;

  static hdf5pp::Type persType() ;

private:
  AcqirisVertV1_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::HorizV1 class
//
struct AcqirisHorizV1_Data {
  double   sampInterval;
  double   delayTime;
  uint32_t nbrSamples;
  uint32_t nbrSegments;
};

struct AcqirisHorizV1 {
public:
  AcqirisHorizV1 () {}
  AcqirisHorizV1 ( const Pds::Acqiris::HorizV1& v ) ;

  static hdf5pp::Type persType() ;

private:
  AcqirisHorizV1_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::TrigV1 class
//
struct AcqirisTrigV1_Data {
  uint32_t trigCoupling;
  uint32_t trigInput;
  uint32_t trigSlope;
  double   trigLevel;
};

struct AcqirisTrigV1 {
public:
  AcqirisTrigV1 () {}
  AcqirisTrigV1 ( const Pds::Acqiris::TrigV1& v ) ;

  static hdf5pp::Type persType() ;

private:
  AcqirisTrigV1_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::ConfigV1 class
//
struct AcqirisConfigV1_Data {
  uint32_t nbrConvertersPerChannel;
  uint32_t channelMask;
  uint32_t nbrChannels;
  uint32_t nbrBanks;
};

struct AcqirisConfigV1 {
public:
  AcqirisConfigV1 () {}
  AcqirisConfigV1 ( const Pds::Acqiris::ConfigV1& v ) ;

  static hdf5pp::Type persType() ;

private:
  AcqirisConfigV1_Data m_data ;
};

// store the object of type Pds::Acqiris::ConfigV1 at specified location
void storeAcqirisConfigV1 ( const Pds::Acqiris::ConfigV1& config, hdf5pp::Group location ) ;

} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISCONFIGV1_H
