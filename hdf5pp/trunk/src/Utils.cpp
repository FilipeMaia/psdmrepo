//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Utils...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/Utils.h"

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

namespace hdf5pp {

// template-free implementation of append()
void
Utils::_append(hdf5pp::Group group, const std::string& dataset, const void* data,
    const Type& native_type, const Type& stored_type)
{
  // open or create rank-1 dataset
  DataSet ds = group.openDataSet(dataset);
  if (not ds.valid()) {
    ds = group.createDataSet(dataset, stored_type, DataSpace::makeSimple(0, H5S_UNLIMITED));
  }

  // get current size
  DataSpace dsp = ds.dataSpace();
  hsize_t size = dsp.size();

  // extend dataset
  ds.set_extent(size+1);

  // store the data in dataset
  ds.store(DataSpace::makeScalar(), dsp.select_single(size), data, native_type);
}

// template-free implementation of storeScalar()
void
Utils::_storeScalar(hdf5pp::Group group, const std::string& dataset, const void* data,
    const Type& native_type, const Type& stored_type)
{
  // create new scalar dataset
  DataSet ds = group.createDataSet(dataset, stored_type, DataSpace::makeScalar());

  // store the data in dataset
  ds.store(DataSpace::makeScalar(), DataSpace::makeScalar(), data, native_type);
}

} // namespace hdf5pp
