#ifndef H5DATATYPES_ARRAYCHARDATAV1_H
#define H5DATATYPES_ARRAYCHARDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ArraycharDataV1.
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
#include "pdsdata/psddl/arraychar.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper class for Pds::Arraychar::DataV1
//
class ArraycharDataV1  {
public:

  typedef Pds::Arraychar::DataV1 XtcType;

  // Default constructor
  ArraycharDataV1() {}
  ArraycharDataV1(const XtcType& data);

  ~ArraycharDataV1();

  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc + xtc.numChars(); }

protected:

private:
  
  uint64_t numChars;
  size_t vlen_data;
  uint8_t* data;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ARRAYCHARDATAV1_H
