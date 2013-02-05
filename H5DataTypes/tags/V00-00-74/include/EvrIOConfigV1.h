#ifndef H5DATATYPES_EVRIOCONFIGV1_H
#define H5DATATYPES_EVRIOCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrIOConfigV1.
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
#include "pdsdata/evr/IOConfigV1.hh"
#include "H5DataTypes/EvrConfigData.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EvrIOConfigV1_Data {
  int16_t conn ;
};

class EvrIOConfigV1  {
public:

  typedef Pds::EvrData::IOConfigV1 XtcType ;

  EvrIOConfigV1 () {}
  EvrIOConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  EvrIOConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRIOCONFIGV1_H
