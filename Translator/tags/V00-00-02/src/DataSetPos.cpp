#include "Translator/DataSetPos.h"
#include "hdf5pp/Exceptions.h"

using namespace Translator;

/*
DataSetPos::DataSetPos(hid_t dsetId) : m_dsetId(dsetId), m_nextWrite(0), m_valid(true)
{
}

void DataSetPos::close() {
}

void DataSetPos::prepareToWrite() {
  hsize_t start = m_nextWrite;
  ++m_nextWrite;
  hsize_t size[m_rankOne];
  size[0] = m_nextWrite;
  herr_t stat = H5Dset_extent(m_dsetId, size);
  if (stat < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "Failed to set_extent in prepareToWrite");
  hsize_t count = 1;
  stat = H5Sselect_hyperslab(m_fileDspaceId, 
                             H5S_SELECT_SET, 
                             &start, 
                             NULL,
                             &count, 
                             NULL);
  if (stat < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "Failed to select hyperslab in prepareToWrite");
}

*/
