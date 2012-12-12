//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5DatasetIter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5DatasetIter.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "PSHdf5Input/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "Hdf5DatasetIter";


  // make data type for time dataset
  hdf5pp::Type initIterDataType()
  {
    hdf5pp::CompoundType dataType = hdf5pp::CompoundType::compoundType<PSHdf5Input::Hdf5DatasetIterData>() ;
    dataType.insert_native<uint32_t>("seconds", offsetof(PSHdf5Input::Hdf5DatasetIterData, sec));
    dataType.insert_native<uint32_t>("nanoseconds", offsetof(PSHdf5Input::Hdf5DatasetIterData, nsec));
    return dataType;
  }

  // data type for in-memory data
  hdf5pp::Type iterDataType = initIterDataType();
}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

// Constructor for "begin" iterator
Hdf5DatasetIter::Hdf5DatasetIter (const hdf5pp::Group& grp, Tag tag)
  : m_ds()
  , m_size(0)
  , m_index(0)
  , m_data()
{
  // fill constant part of data object
  m_data.group = grp;
  
  // open "time" dataset, will throw if cannot open
  m_ds = m_data.group.openDataSet<Hdf5DatasetIterData>("time");
  
  // get dataset's dataspace
  m_dsp = m_ds.dataSpace();
  
  // check rank
  if (m_dsp.rank() != 1) {
    throw FileStructure(ERR_LOC, "bad rank for 'time' dataset in group "+grp.name());
  }

  // get its size
  m_size = m_dsp.size();
  MsgLog(logger, debug, "dataset size: " << m_size << " in group " << grp.name());

  // move past end for end iterator
  if (tag == End) m_index = m_size;
  
  // update object from dataset
  updateData();
}

// read m_data from dataset if possible
void 
Hdf5DatasetIter::updateData()
{
  m_data.index = m_index;
  
  // if past end then do nothing
  if (m_index < 0 or uint64_t(m_index) >= m_size) return;
  
  // in-memory dataspace
  hdf5pp::DataSpace mem_ds = hdf5pp::DataSpace::makeScalar();

  // define hyper-slab for in-file dataspace
  hsize_t offset[1] = { m_index };
  hsize_t count[1] = { 1 };
  m_dsp.select_hyperslab(H5S_SELECT_SET, offset, 0, count, 0);
  
  m_ds.read(mem_ds, m_dsp, &m_data, ::iterDataType);
}

} // namespace PSHdf5Input
