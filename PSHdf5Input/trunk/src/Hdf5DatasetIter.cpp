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
  hdf5pp::Type iterDataType(bool fullTsFormat)
  {
    hdf5pp::CompoundType dataType = hdf5pp::CompoundType::compoundType<PSHdf5Input::Hdf5DatasetIterData>() ;
    dataType.insert_native<uint32_t>("seconds", offsetof(PSHdf5Input::Hdf5DatasetIterData, sec));
    dataType.insert_native<uint32_t>("nanoseconds", offsetof(PSHdf5Input::Hdf5DatasetIterData, nsec));
    if (fullTsFormat) {
      dataType.insert_native<uint32_t>("ticks", offsetof(PSHdf5Input::Hdf5DatasetIterData, ticks));
      dataType.insert_native<uint32_t>("fiducials", offsetof(PSHdf5Input::Hdf5DatasetIterData, fiducials));
      dataType.insert_native<uint32_t>("control", offsetof(PSHdf5Input::Hdf5DatasetIterData, control));
      dataType.insert_native<uint32_t>("vector", offsetof(PSHdf5Input::Hdf5DatasetIterData, vector));
    }
    return dataType;
  }
}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

// Constructor for "begin" iterator
Hdf5DatasetIter::Hdf5DatasetIter (const hdf5pp::Group& grp, bool fullTsFormat, Tag tag)
  : m_group(grp)
  , m_fullTsFormat(fullTsFormat)
  , m_ds()
  , m_size(0)
  , m_index(0)
  , m_dataIndex(0)
  , m_data()
{
  // open "time" dataset, will throw if cannot open
  m_ds = m_group.openDataSet("time");
  
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
  // if past end then do nothing
  if (m_index < 0 or uint64_t(m_index) >= m_size) return;

  // if data for the index already there do nothing
  if (uint64_t(m_index) >= m_dataIndex and uint64_t(m_index) < m_dataIndex + m_data.size()) return;

  // determine how many items to read and where to start
  size_t size = 512;
  m_dataIndex = (m_index/size)*size;
  if (m_dataIndex + size > m_size) size = m_size - m_dataIndex;

  // allocate space
  m_data.resize(size);

  // set  parts of data that are not read from file
  for (unsigned i = 0; i != size; ++ i) {
    m_data[i].index = m_dataIndex + i;
    m_data[i].group = m_group;
  }
  
  // in-memory dataspace
  hdf5pp::DataSpace mem_ds = hdf5pp::DataSpace::makeSimple(size, size);

  // define hyper-slab for in-file dataspace
  hsize_t start[] = { m_dataIndex };
  hsize_t count[] = { size };
  m_dsp.select_hyperslab(H5S_SELECT_SET, start, 0, count, 0);
  
  m_ds.read(mem_ds, m_dsp, m_data.data(), ::iterDataType(m_fullTsFormat));
}

} // namespace PSHdf5Input
