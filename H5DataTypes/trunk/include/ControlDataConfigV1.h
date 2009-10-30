#ifndef H5DATATYPES_CONTROLDATACONFIGV1_H
#define H5DATATYPES_CONTROLDATACONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlDataConfigV1.
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
#include "pdsdata/control/ConfigV1.hh"
#include "pdsdata/control/PVControl.hh"
#include "pdsdata/control/PVMonitor.hh"

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
struct ControlDataPVControlV1_Data {
  char     name[Pds::ControlData::PVControl::NameSize];
  uint32_t index;
  double   value;
} ;

class ControlDataPVControlV1 {
public:

  ControlDataPVControlV1 () {}
  ControlDataPVControlV1 ( const Pds::ControlData::PVControl& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  ControlDataPVControlV1_Data m_data ;
};


//
// Helper type for Pds::ControlData::PVMonitor
//
struct ControlDataPVMonitorV1_Data {
  char     name[Pds::ControlData::PVMonitor::NameSize];
  uint32_t index;
  double   loValue;
  double   hiValue;
} ;

class ControlDataPVMonitorV1 {
public:

  ControlDataPVMonitorV1 () {}
  ControlDataPVMonitorV1 ( const Pds::ControlData::PVMonitor& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  ControlDataPVMonitorV1_Data m_data ;
};

//
// Helper type for Pds::ControlData::ConfigV1
//
struct ControlDataConfigV1_Data {
  uint8_t   uses_duration;
  uint8_t   uses_events;
  XtcClockTime_Data duration;
  uint32_t  events;
  uint32_t  npvControls;
  uint32_t  npvMonitors;
};

class ControlDataConfigV1  {
public:

  typedef Pds::ControlData::ConfigV1 XtcType ;

  ControlDataConfigV1 () {}
  ControlDataConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  ControlDataConfigV1_Data m_data ;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_CONTROLDATACONFIGV1_H
