#ifndef H5DATATYPES_ACQIRISTDCCONFIGV1_H
#define H5DATATYPES_ACQIRISTDCCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcConfigV1.
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
//  Helper class for Pds::Acqiris::TdcChannel class
//
struct AcqirisTdcChannel {
public:
  AcqirisTdcChannel () {}
  AcqirisTdcChannel ( const Pds::Acqiris::TdcChannel& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  int32_t channel;
  uint16_t mode;
  uint16_t slope;
  double   level;
};

//
//  Helper class for Pds::Acqiris::TdcAuxIO class
//
struct AcqirisTdcAuxIO {
public:
  AcqirisTdcAuxIO () {}
  AcqirisTdcAuxIO ( const Pds::Acqiris::TdcAuxIO& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint16_t channel;
  uint16_t mode;
  uint16_t term;
};

//
//  Helper class for Pds::Acqiris::TdcVetoIO class
//
struct AcqirisTdcVetoIO {
public:
  AcqirisTdcVetoIO () {}
  AcqirisTdcVetoIO ( const Pds::Acqiris::TdcVetoIO& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint16_t channel;
  uint16_t mode;
  uint16_t term;
};

//
//  Helper class for Pds::Acqiris::TdcConfigV1 class
//
struct AcqirisTdcConfigV1 {
public:

  typedef Pds::Acqiris::TdcConfigV1 XtcType ;

  AcqirisTdcConfigV1 () {}
  AcqirisTdcConfigV1(const XtcType&) {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
};


} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISTDCCONFIGV1_H
