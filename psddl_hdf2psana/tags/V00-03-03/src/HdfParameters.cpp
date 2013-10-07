//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfParameters...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/HdfParameters.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_hdf2psana {

// Method that calculates chunk cache size for given storage type.
unsigned
HdfParameters::chunkCacheSize(const hdf5pp::Type& type, unsigned chunk_size)
{
  // chunk cache target size is between 1MB and 10MB but at least 1 full chunk
  const hsize_t def_chunk_cache_size = 1024*1024;
  const hsize_t max_chunk_cache_size = 10*1024*1024;

  hsize_t chunk_size_bytes = chunk_size * type.size();
  hsize_t chunk_cache_size = 1;
  if (chunk_size_bytes <= def_chunk_cache_size/2) {
    chunk_cache_size = def_chunk_cache_size/chunk_size_bytes;
  } else if (chunk_size_bytes <= max_chunk_cache_size/2) {
    chunk_cache_size *= 2;
  }
  return chunk_cache_size;
}

} // namespace psddl_hdf2psana
