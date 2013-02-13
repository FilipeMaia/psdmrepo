#ifndef H5DATATYPES_FEXAMPCONFIGV1_H
#define H5DATATYPES_FEXAMPCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FexampConfigV1.
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
#include "pdsdata/fexamp/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {


//
// Helper class for Pds::Fexamp::ChannelV1
//
class FexampChannelV1 {
public:

  typedef Pds::Fexamp::ChannelV1 XtcType ;

  enum {NValues = XtcType::NumberOfChannelBitFields};

  FexampChannelV1() {}
  FexampChannelV1(const XtcType& data);

  static hdf5pp::Type stored_type() { return native_type(); }
  static hdf5pp::Type native_type() ;

private:
  
  uint32_t m_values[NValues];

};

//
// Helper class for Pds::Fexamp::ASIC_V1
//
class FexampASIC_V1 {
public:

  typedef Pds::Fexamp::ASIC_V1 XtcType ;

  enum {NValues = XtcType::NumberOfASIC_Entries};

  FexampASIC_V1() {}
  FexampASIC_V1(const XtcType& data);

  static hdf5pp::Type stored_type() { return native_type(); }
  static hdf5pp::Type native_type() ;

private:
  
  uint32_t m_values[NValues];

};

//
// Helper class for Pds::Fexamp::ConfigV1
//
class FexampConfigV1  {
public:

  typedef Pds::Fexamp::ConfigV1 XtcType ;

  enum {NValues = XtcType::NumberOfRegisters};

  FexampConfigV1() {}
  FexampConfigV1(const XtcType& data);

  static hdf5pp::Type stored_type() { return native_type(); }
  static hdf5pp::Type native_type() ;

  static void store(const XtcType& config, hdf5pp::Group location);

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  
  uint32_t m_FPGAversion;
  uint32_t m_values[NValues];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_FEXAMPCONFIGV1_H
