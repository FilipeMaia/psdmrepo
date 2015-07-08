#ifndef TRANSLATOR_HDFWRITERNEW_H
#define TRANSLATOR_HDFWRITERNEW_H

#include <typeinfo>
#include <string>
#include "hdf5/hdf5.h"

namespace Translator {

class HdfWriterNew {
 public:

  typedef hid_t (*CreateHDF5Type)(const void *userDataType);
  typedef const void * (*FillHdf5WriteBuffer)(const void *userDataType);
  typedef void (*CloseHDF5Type)(hid_t typeId);

 HdfWriterNew(const std::type_info *typeInfoPtr,
              std::string datasetName, CreateHDF5Type createType, 
              FillHdf5WriteBuffer fillWriteBuffer, CloseHDF5Type closeType=NULL)
   : m_typeInfoPtr(typeInfoPtr),
     m_datasetName(datasetName),
     m_createType(createType),
     m_fillWriteBuffer(fillWriteBuffer), 
     m_closeType(closeType)  {}

  const std::type_info *typeInfoPtr() const { return m_typeInfoPtr; }
  const std::string & datasetName() const { return m_datasetName; }
  CreateHDF5Type createType()  const { return m_createType; }
  FillHdf5WriteBuffer fillWriteBuffer()  const { return m_fillWriteBuffer; }
  CloseHDF5Type closeType()  const { return m_closeType; }
 private:
  const std::type_info *m_typeInfoPtr;
  std::string m_datasetName;
  CreateHDF5Type m_createType;
  FillHdf5WriteBuffer m_fillWriteBuffer;
  CloseHDF5Type m_closeType;
}; // HdfNewWriter

} // namesapce Translator

#endif
