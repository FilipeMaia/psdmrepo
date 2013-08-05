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

// Create a chunked rank=1 dataset.
DataSet
Utils::createDataset(hdf5pp::Group group, const std::string& dataset, const Type& stored_type,
    hsize_t chunk_size, hsize_t chunk_cache_size, int deflate, bool shuffle)
{
  // make extensible data space
  hdf5pp::DataSpace dsp = hdf5pp::DataSpace::makeSimple(0, H5S_UNLIMITED);

  // use chunking
  hdf5pp::PListDataSetCreate plDScreate ;
  plDScreate.set_chunk(chunk_size) ;

  // optionally set filters
  if (shuffle) plDScreate.set_shuffle() ;
  if (deflate >= 0) plDScreate.set_deflate(deflate) ;

  // set chunk cache parameters
  hdf5pp::PListDataSetAccess plDSaccess;
  size_t chunk_cache_bytes = chunk_cache_size * chunk_size * stored_type.size();
  // ideally this number should be prime, can live with non-prime for now
  size_t rdcc_nslots = chunk_cache_size*19;
  plDSaccess.set_chunk_cache(rdcc_nslots, chunk_cache_bytes);

  // make a data set
  return group.createDataSet(dataset, stored_type, dsp, plDScreate, plDSaccess);
}

// template-free implementation of append()
void
Utils::_append(hdf5pp::Group group, const std::string& dataset, const void* data,
    const Type& native_type, const Type& stored_type)
{
  // open or create rank-1 dataset
  DataSet ds = group.openDataSet(dataset);

  // get current size
  DataSpace dsp = ds.dataSpace();
  hsize_t size = dsp.size();

  // extend dataset
  ds.set_extent(size+1);

  // get updated dataspace
  dsp = ds.dataSpace();

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

// template-free implementation of storeArray()
void
Utils::_storeArray(hdf5pp::Group group, const std::string& dataset, const void* data,
    unsigned rank, const unsigned* shape, const Type& native_type, const Type& stored_type)
{
  std::vector<hsize_t> dims(shape, shape+rank);

  // create new dataspace
  DataSpace dsp = DataSpace::makeSimple(rank, &dims.front(), &dims.front());
  DataSet ds = group.createDataSet(dataset, stored_type, dsp);

  // store the data in dataset
  ds.store(dsp, dsp, data, native_type);
}

} // namespace hdf5pp
