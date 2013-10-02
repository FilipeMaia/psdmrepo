#ifndef H5DATATYPES_L3TCONFIGV1_H
#define H5DATATYPES_L3TCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L3TConfigV1.
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
#include "pdsdata/psddl/l3t.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::L3T::ConfigV1
//
class L3TConfigV1  {
public:

  typedef Pds::L3T::ConfigV1 XtcType ;

  L3TConfigV1 () : module_id(0), description(0) {}
  L3TConfigV1 ( const XtcType& data ) ;

  ~L3TConfigV1 ();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

protected:

private:

  uint32_t module_id_len;  // Not needed for HDF5, but will make it easier to read data back
  uint32_t desc_len;       // Not needed for HDF5, but will make it easier to read data back
  char* module_id;
  char* description;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_L3TCONFIGV1_H
