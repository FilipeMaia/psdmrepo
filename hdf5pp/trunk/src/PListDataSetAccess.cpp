//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListDataSetAccess...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/PListDataSetAccess.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

//----------------
// Constructors --
//----------------
PListDataSetAccess::PListDataSetAccess ()
  : m_impl(H5P_DATASET_ACCESS)
{
}

//--------------
// Destructor --
//--------------
PListDataSetAccess::~PListDataSetAccess ()
{
}

// define chuk cache properties
void
PListDataSetAccess::set_chunk_cache(size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0)
{
  herr_t stat = H5Pset_chunk_cache(m_impl.id(), rdcc_nslots, rdcc_nbytes, rdcc_w0);
  if (stat < 0) {
    throw Hdf5CallException("PListDataSetCreate::set_chunk", "H5Pset_chunk");
  }
}

} // namespace hdf5pp
