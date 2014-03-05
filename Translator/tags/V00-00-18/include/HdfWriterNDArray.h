#include <string>
#include "ndarray/ndarray.h"
#include "PSEvt/EventKey.h"
#include "PSEnv/Env.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterGeneric.h"
#include "Translator/HdfWriterFromEvent.h"

namespace Translator {

// ----------------------------------------
// C++ basic base types for NDarrays to HDF5 type ids:

template<class ElemType>
hid_t getH5BaseType() { throw ErrSvc::Issue(ERR_LOC, "unsupported base type for hdf5"); }

template<>
hid_t getH5BaseType<uint8_t>() { return H5T_NATIVE_UINT8; }

template<>
hid_t getH5BaseType<uint16_t>() { return H5T_NATIVE_UINT16; }

template<>
hid_t getH5BaseType<uint32_t>() { return H5T_NATIVE_UINT32; }

template<>
hid_t getH5BaseType<uint64_t>() { return H5T_NATIVE_UINT64; }

template<>
hid_t getH5BaseType<int8_t>() { return H5T_NATIVE_INT8; }

template<>
hid_t getH5BaseType<int16_t>() { return H5T_NATIVE_INT16; }

template<>
hid_t getH5BaseType<int32_t>() { return H5T_NATIVE_INT32; }

template<>
hid_t getH5BaseType<int64_t>() { return H5T_NATIVE_INT64; }

template<>
hid_t getH5BaseType<float>() { return H5T_NATIVE_FLOAT; }

template<>
hid_t getH5BaseType<double>() { return H5T_NATIVE_DOUBLE; }

// --------------------------------------------------
// Determine if a NDArray has C or Fortran strides, throw exception if neither
template<class ElemType, unsigned NDim>
enum ndns::Order get_C_orFortran_StrideOnly(const ndarray<ElemType, NDim> &array, const std::string & errorMsg) {
  if (NDim==0 or NDim==1) return ndns::C;
  int C_strides[NDim];
  int Fortran_strides[NDim];
  C_strides[NDim-1]=1;
  Fortran_strides[0]=1;
  for (int i = int(NDim)-2; i >= 0; --i) C_strides[i] = C_strides[i+1] * array.shape()[i+1];
  for (unsigned i = 1; i < NDim; ++i) Fortran_strides[i] = Fortran_strides[i-1] * array.shape()[i-1];

  bool isC=true;
  bool isFortran = true;
  for (unsigned dim = 0; dim < NDim; ++dim) {
    if (array.strides()[dim] != C_strides[dim]) isC = false;
    if (array.strides()[dim] != Fortran_strides[dim]) isFortran = false;
  }
  if (isC) return ndns::C;
  if (isFortran) return ndns::Fortran;
  throw ErrSvc::Issue(ERR_LOC, errorMsg);
}

//----------------------------------------------
// NDArray writer:
template <class ElemType, unsigned NDim>
class HdfWriterNDArray : public HdfWriterFromEvent {
 public:
 HdfWriterNDArray() : m_writer("ndarray") {}
  void make_datasets(DataTypeLoc dataTypeLoc, hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, PSEnv::Env & env,
                     bool shuffle, int deflate,
                     boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy) 
  {
    checkType<ndarray<ElemType,NDim> >(eventKey, "HdfWriterNDArray");
    boost::shared_ptr< ndarray<ElemType, NDim> > ndarrayPtr;
    if (dataTypeLoc == inEvent) ndarrayPtr = evt.get(eventKey.src(), eventKey.key()); 
    else if (dataTypeLoc == inConfigStore) ndarrayPtr = env.configStore().get(eventKey.src());

    enum ndns::Order order = get_C_orFortran_StrideOnly<ElemType,NDim>(*ndarrayPtr, errorMsgForUnsupportedStride);
    if (order != ndns::C) throw NotImplementedException(ERR_LOC, "Fortran stride not implemented");

    NDArrayFormat ndArrayFormat;
    hsize_t dims[NDim];
    ndArrayFormat.order = order;
    for (unsigned i = 0; i < NDim; ++i) {
      ndArrayFormat.dim[i]=ndarrayPtr->shape()[i];
      dims[i]=ndarrayPtr->shape()[i];
    }
    hid_t baseType = getH5BaseType<ElemType>();
    hid_t arrayType = H5Tarray_create2(baseType,NDim, dims);
    
    ndArrayFormat.arrayHdf5BaseType = baseType;
    ndArrayFormat.arrayHdf5Type = arrayType;
    m_firstArrayOfDatasetFormat[eventKey] = ndArrayFormat;
    
    Translator::DataSetCreationProperties dataSetCreationProperties(chunkPolicy,shuffle, deflate);
    m_writer.createUnlimitedSizeDataset(srcGroup.id(), "data", arrayType, dataSetCreationProperties);
  }

  void append(DataTypeLoc dataTypeLoc,
              hdf5pp::Group & srcGroup, const PSEvt::EventKey & eventKey, 
              PSEvt::Event & evt, PSEnv::Env & env) 
  {
    MsgLog("HdfWriterNDArray",debug,"HdfWriterNDArray::append");
    checkType<ndarray<ElemType,NDim> >(eventKey, "HdfWriterNDArray");
    
    boost::shared_ptr< ndarray<ElemType,NDim> > ptr; 
    if (dataTypeLoc == inEvent) ptr = evt.get(eventKey.src(), eventKey.key()); 
    else if (dataTypeLoc == inConfigStore) ptr = env.configStore().get(eventKey.src());
    
    if (m_firstArrayOfDatasetFormat.find(eventKey) == m_firstArrayOfDatasetFormat.end()) {
      throw ErrSvc::Issue(ERR_LOC, "HDFWriterNDArray::append called for unknown eventKey");
    }

    const NDArrayFormat & firstNdArrayFormat = m_firstArrayOfDatasetFormat[eventKey];
    
    if (firstNdArrayFormat.order != get_C_orFortran_StrideOnly(*ptr,errorMsgForUnsupportedStride)) {
      throw ErrSvc::Issue(ERR_LOC, "HdfWriterNDArray::append, this array has an order (C or Fortran) different from the first array in the dataset");
    }
    bool sameDimensionsAtFirstNDarray = true;
    for (unsigned i = 0; i < NDim; ++i) {
      if (firstNdArrayFormat.dim[i] != ptr->shape()[i]) sameDimensionsAtFirstNDarray = false;
    }
    if (not sameDimensionsAtFirstNDarray) {
      throw ErrSvc::Issue(ERR_LOC, "HdfWriterNDArray::append, this array has dimensions different from the first array in the dataset");
    }
    m_writer.append(srcGroup.id(), "data", ptr->data());
  }
  
  void store(DataTypeLoc dataTypeLoc, 
             hdf5pp::Group & srcGroup, 
             const PSEvt::EventKey & eventKey, 
             PSEvt::Event & evt, 
             PSEnv::Env & env) { throw NotImplementedException(ERR_LOC, "store()"); }

  void store_at(DataTypeLoc dataTypeLoc, 
                long index, hdf5pp::Group & srcGroup, 
                const PSEvt::EventKey & eventKey, 
                PSEvt::Event & evt, 
                PSEnv::Env & env) { throw NotImplementedException(ERR_LOC, "store_at()"); }

  void addBlank(hdf5pp::Group & group) { throw NotImplementedException(ERR_LOC, "addBlank()"); }

  void closeDatasets(hdf5pp::Group & group) { m_writer.closeDatasets(group.id()); }

  class NotImplementedException : public ErrSvc::Issue {
  public:
  NotImplementedException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
  }; // class NotImplementedException
 private:
  struct NDArrayFormat {
    unsigned dim[NDim];
    enum ndns::Order order;
    hid_t arrayHdf5BaseType;
    hid_t arrayHdf5Type;
  }; // struct NDArrayFormat

  Translator::HdfWriterGeneric m_writer;

  std::map< PSEvt::EventKey , NDArrayFormat > m_firstArrayOfDatasetFormat;

  static const std::string errorMsgForUnsupportedStride;
  
}; // HdfWriterNDArray

} // namespace Translator

template <class ElemType, unsigned NDim>
  const std::string Translator::HdfWriterNDArray<ElemType, NDim>::errorMsgForUnsupportedStride("HdfWriterNDArray::make_datasets, only C or Fortran strides supported");

