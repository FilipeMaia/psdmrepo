#ifndef H5DATATYPES_GSC16AIDATAV1_H
#define H5DATATYPES_GSC16AIDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiDataV1.
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
#include "pdsdata/gsc16ai/ConfigV1.hh"
#include "pdsdata/gsc16ai/DataV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper class for Pds::Gsc16ai::DataV1
//
class Gsc16aiDataV1  {
public:

  typedef Pds::Gsc16ai::DataV1 XtcType;
  typedef Pds::Gsc16ai::ConfigV1 ConfigXtcType;

  // Default constructor
  Gsc16aiDataV1() {}
  Gsc16aiDataV1(const XtcType& data);

  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();

  static hdf5pp::Type stored_data_type(const ConfigXtcType& config);

protected:

private:

  enum {NTimestamps = 3};
  
  uint16_t _timestamp[NTimestamps];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_GSC16AIDATAV1_H
