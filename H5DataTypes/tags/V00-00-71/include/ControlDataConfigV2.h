#ifndef H5DATATYPES_CONTROLDATACONFIGV2_H
#define H5DATATYPES_CONTROLDATACONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV2.
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
#include "H5DataTypes/XtcClockTime.h"
#include "hdf5pp/Group.h"
#include "pdsdata/control/ConfigV2.hh"
#include "pdsdata/control/PVLabel.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::ControlData::PVControl
//
class ControlDataPVLabelV1 {
public:

  ControlDataPVLabelV1 () {}
  ControlDataPVLabelV1 ( const Pds::ControlData::PVLabel& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  char     name[Pds::ControlData::PVLabel::NameSize];
  char     value[Pds::ControlData::PVLabel::ValueSize];
};


//
// Helper type for Pds::ControlData::ConfigV2
//
class ControlDataConfigV2  {
public:

  typedef Pds::ControlData::ConfigV2 XtcType ;

  ControlDataConfigV2 () {}
  ControlDataConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  uint8_t   uses_duration;
  uint8_t   uses_events;
  XtcClockTime duration;
  uint32_t  events;
  uint32_t  npvControls;
  uint32_t  npvMonitors;
  uint32_t  npvLabels;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CONTROLDATACONFIGV2_H
