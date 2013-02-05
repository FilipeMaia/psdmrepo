#ifndef H5DATATYPES_EPICSCONFIGV1_H
#define H5DATATYPES_EPICSCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsConfigV1.
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
#include "pdsdata/epics/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Epics::PvConfigV1
//
class EpicsPvConfigV1 {
public:

  EpicsPvConfigV1() {}
  EpicsPvConfigV1(const Pds::Epics::PvConfigV1& pvConfig);

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  enum { iMaxPvDescLength = Pds::Epics::PvConfigV1::iMaxPvDescLength };

  int16_t pvId;
  char    description[iMaxPvDescLength];
  float   interval;
};


//
// Helper type for Pds::Epics::ConfigV1
//
class EpicsConfigV1  {
public:

  typedef Pds::Epics::ConfigV1 XtcType ;

  EpicsConfigV1() {}
  EpicsConfigV1(const XtcType& data);

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store(const XtcType& config, hdf5pp::Group location);

  static size_t xtcSize(const XtcType& xtc) { return xtc.size(); }

private:

  uint32_t  numPv;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPICSCONFIGV1_H
