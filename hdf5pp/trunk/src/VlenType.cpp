//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class VlenType...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/VlenType.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

// make a an array type of any rank
VlenType
VlenType::vlenType( const Type& baseType )
{
  hid_t tid = H5Tvlen_create( baseType.id() ) ;
  if ( tid < 0 ) throw Hdf5CallException ( "VlenType::vlenType", "H5Tvlen_create" ) ;
  return VlenType ( tid ) ;
}

} // namespace hdf5pp
