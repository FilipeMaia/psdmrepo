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
#include "pdsdata/acqiris/TdcConfigV1.hh"

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
//  Helper classes for Pds::Acqiris::TdcChannel class
//
struct AcqirisTdcChannel_Data {
  int32_t channel;
  uint16_t mode;
  uint16_t slope;
  double   level;
};

struct AcqirisTdcChannel {
public:
  AcqirisTdcChannel () {}
  AcqirisTdcChannel ( const Pds::Acqiris::TdcChannel& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  AcqirisTdcChannel_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::TdcAuxIO class
//
struct AcqirisTdcAuxIO_Data {
  uint16_t channel;
  uint16_t mode;
  uint16_t term;
};

struct AcqirisTdcAuxIO {
public:
  AcqirisTdcAuxIO () {}
  AcqirisTdcAuxIO ( const Pds::Acqiris::TdcAuxIO& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  AcqirisTdcAuxIO_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::TdcVetoIO class
//
struct AcqirisTdcVetoIO_Data {
  uint16_t channel;
  uint16_t mode;
  uint16_t term;
};

struct AcqirisTdcVetoIO {
public:
  AcqirisTdcVetoIO () {}
  AcqirisTdcVetoIO ( const Pds::Acqiris::TdcVetoIO& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  AcqirisTdcVetoIO_Data m_data ;
};

//
//  Helper classes for Pds::Acqiris::TdcConfigV1 class
//
struct AcqirisTdcConfigV1 {
public:

  typedef Pds::Acqiris::TdcConfigV1 XtcType ;

  AcqirisTdcConfigV1 () {}

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
};


} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISTDCCONFIGV1_H
