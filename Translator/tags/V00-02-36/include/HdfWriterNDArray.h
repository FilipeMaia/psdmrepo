
#include <sstream>
#include <string>
#include "ndarray/ndarray.h"
#include "PSEvt/EventKey.h"
#include "PSEnv/Env.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterGeneric.h"
#include "Translator/HdfWriterFromEvent.h"
#include "Translator/specialKeyStrings.h"

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

template<>
hid_t getH5BaseType<long double>() { return H5T_NATIVE_LDOUBLE; }

template<>
hid_t getH5BaseType<const uint8_t>() { return H5T_NATIVE_UINT8; }

template<>
hid_t getH5BaseType<const uint16_t>() { return H5T_NATIVE_UINT16; }

template<>
hid_t getH5BaseType<const uint32_t>() { return H5T_NATIVE_UINT32; }

template<>
hid_t getH5BaseType<const uint64_t>() { return H5T_NATIVE_UINT64; }

template<>
hid_t getH5BaseType<const int8_t>() { return H5T_NATIVE_INT8; }

template<>
hid_t getH5BaseType<const int16_t>() { return H5T_NATIVE_INT16; }

template<>
hid_t getH5BaseType<const int32_t>() { return H5T_NATIVE_INT32; }

template<>
hid_t getH5BaseType<const int64_t>() { return H5T_NATIVE_INT64; }

template<>
hid_t getH5BaseType<const float>() { return H5T_NATIVE_FLOAT; }

template<>
hid_t getH5BaseType<const double>() { return H5T_NATIVE_DOUBLE; }

template<>
hid_t getH5BaseType<const long double>() { return H5T_NATIVE_LDOUBLE; }

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

template<class ElemType, unsigned NDim>
boost::shared_ptr< ndarray< ElemType, NDim> > 
getNDArrayPtr(const PSEvt::EventKey &eventKey, const DataTypeLoc dataTypeLoc, 
              PSEvt::Event &evt, PSEnv::Env & env) 
{
  checkType<ndarray<ElemType,NDim> >(eventKey, "HdfWriterNDArray");
  boost::shared_ptr< ndarray< ElemType, NDim> > ndarrayPtr;
  if (dataTypeLoc == inEvent) ndarrayPtr = evt.get(eventKey.src(), eventKey.key()); 
  else if (dataTypeLoc == inConfigStore) ndarrayPtr = env.configStore().get(eventKey.src(), eventKey.key());
  else if (dataTypeLoc == inCalibStore) ndarrayPtr = env.calibStore().get(eventKey.src(), eventKey.key());
  return ndarrayPtr;
}

//----------------------------------------------
// NDArray writer:
 template <class ElemType, unsigned NDim, bool vlen>
class HdfWriterNDArray : public HdfWriterFromEvent {
 public:
 HdfWriterNDArray() : m_writer(std::string("ndarray")+ (vlen ? std::string("Vlen"): std::string(""))) {}
  void make_datasets(DataTypeLoc dataTypeLoc, hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, PSEnv::Env & env,
                     bool shuffle, int deflate,
                     boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy) 
  {
    boost::shared_ptr< ndarray<ElemType, NDim> > ndarrayPtr;
    ndarrayPtr= getNDArrayPtr<ElemType, NDim>(eventKey, dataTypeLoc, evt, env);
    if (not ndarrayPtr) throw EventKeyNotFound(ERR_LOC, eventKey, dataTypeLoc);

    enum ndns::Order order = get_C_orFortran_StrideOnly<ElemType,NDim>(*ndarrayPtr, errorMsgForUnsupportedStride);
    if (order != ndns::C) throw NotImplementedException(ERR_LOC, "Fortran stride not implemented");

    NDArrayFormat ndArrayFormat;
    hsize_t dims[NDim];
    ndArrayFormat.order = order;
    bool hasZeroDim = false;
    for (unsigned i = 0; i < NDim; ++i) {
      ndArrayFormat.dim[i]=ndarrayPtr->shape()[i];
      dims[i]=ndarrayPtr->shape()[i];
      if (dims[i]==0) hasZeroDim = true;
    }
    if (hasZeroDim) throw NotImplementedException(ERR_LOC, "Arrays with a 0 in the shape are not supported");
    hid_t baseType = getH5BaseType<ElemType>();
    hid_t arrayType;
    hid_t vlenType = -1;
    if (vlen) {
      if (NDim == 1) {
        arrayType = baseType;
      } else {
        if (order != ndns::C) throw ErrSvc::Issue(ERR_LOC, "vlen arrays only supported for C stride - slow dim must be dim0");
        arrayType = H5Tarray_create2(baseType,NDim-1, &dims[1]);
        if (arrayType < 0) throw ErrSvc::Issue(ERR_LOC,"H5Tarray_create2 call failed");
      }
      vlenType = H5Tvlen_create(arrayType);
      if (vlenType < 0) throw ErrSvc::Issue(ERR_LOC, "H5Tvlen_create call failed");
    } else {
      arrayType = H5Tarray_create2(baseType,NDim, dims);
      if (arrayType < 0) throw ErrSvc::Issue(ERR_LOC, "H5Tarray_create2 call failed");
    }
    
    ndArrayFormat.arrayHdf5BaseType = baseType;
    ndArrayFormat.arrayHdf5Type = arrayType;
    m_firstArrayOfDatasetFormat[eventKey] = ndArrayFormat;
    
    Translator::DataSetCreationProperties dataSetCreationProperties(chunkPolicy, shuffle, deflate);
    if (vlen) {
      m_writer.createUnlimitedSizeDataset(srcGroup.id(), "data", vlenType, vlenType, dataSetCreationProperties);
    } else {
      m_writer.createUnlimitedSizeDataset(srcGroup.id(), "data", arrayType, arrayType, dataSetCreationProperties);
    }
  }

  void append(DataTypeLoc dataTypeLoc,
              hdf5pp::Group & srcGroup, const PSEvt::EventKey & eventKey, 
              PSEvt::Event & evt, PSEnv::Env & env) 
  {
    MsgLog("HdfWriterNDArray",debug,"HdfWriterNDArray::append for " << eventKey);
    boost::shared_ptr< ndarray<ElemType, NDim> > ndarrayPtr;
    ndarrayPtr= getNDArrayPtr<ElemType, NDim>(eventKey, dataTypeLoc, evt, env);
    if (not ndarrayPtr) throw EventKeyNotFound(ERR_LOC, eventKey, dataTypeLoc);
    
    if (m_firstArrayOfDatasetFormat.find(eventKey) == m_firstArrayOfDatasetFormat.end()) {
      throw ErrSvc::Issue(ERR_LOC, "HDFWriterNDArray::append called for unknown eventKey");
    }

    const NDArrayFormat & firstNdArrayFormat = m_firstArrayOfDatasetFormat[eventKey];
    
    if (firstNdArrayFormat.order != get_C_orFortran_StrideOnly(*ndarrayPtr,errorMsgForUnsupportedStride)) {
      throw ErrSvc::Issue(ERR_LOC, "HdfWriterNDArray::append, this array has an order (C or Fortran) different from the first array in the dataset");
    }
    bool sameDimensionsAtFirstNDarray = true;
    unsigned startDim = 0;
    if (vlen) startDim = 1;
    for (unsigned i = startDim; i < NDim; ++i) {
      if (firstNdArrayFormat.dim[i] != ndarrayPtr->shape()[i]) sameDimensionsAtFirstNDarray = false;
    }
    if (not sameDimensionsAtFirstNDarray) {
      if (vlen) {
        std::stringstream str;
        str << "vlen HdfWriterNDArray::append, the fast dimensions of this "
            << "array (all but dim 0) are different from the first array in "
            << "the dataset. They must be the same.";
        throw ErrSvc::Issue(ERR_LOC, str.str());
      } else {
        std::stringstream str;
        str << "HdfWriterNDArray::append, this array has dimensions different "
            << "from the first array in the dataset. To write variable length ndarrays "
            << "to the same data set, prepend '" << ndarrayVlenPrefix() 
            << ":' to key. Such ndarrays may only vary in slow dimension.";
        throw ErrSvc::Issue(ERR_LOC, str.str());
      }
    }
    void * data=0;
    hvl_t vdata;
    vdata.len=0;
    vdata.p=0;
    if (vlen) {
      unsigned slowLength = ndarrayPtr->shape()[0];
      vdata.len = slowLength;
      vdata.p = (void *)ndarrayPtr->data();
      data = &vdata;
    } else {
      data = (void *)ndarrayPtr->data();
    }
    m_writer.append(srcGroup.id(), "data", data);
  }
  
  void store(DataTypeLoc dataTypeLoc, 
             hdf5pp::Group & srcGroup, 
             const PSEvt::EventKey & eventKey, 
             PSEvt::Event & evt, 
             PSEnv::Env & env) 
  {
    boost::shared_ptr< ndarray<ElemType, NDim> > ndarrayPtr;
    ndarrayPtr= getNDArrayPtr<ElemType, NDim>(eventKey, dataTypeLoc, evt, env);
    if (not ndarrayPtr) throw EventKeyNotFound(ERR_LOC, eventKey, dataTypeLoc);

    enum ndns::Order order = get_C_orFortran_StrideOnly<ElemType,NDim>(*ndarrayPtr, errorMsgForUnsupportedStride);
    if (order != ndns::C) throw NotImplementedException(ERR_LOC, "Fortran stride not implemented");

    hsize_t dims[NDim];
    bool hasZeroDim = false;
    for (unsigned i = 0; i < NDim; ++i) {
      dims[i]=ndarrayPtr->shape()[i];
      if (dims[i]==0) hasZeroDim = true;
    }
    if (hasZeroDim) throw NotImplementedException(ERR_LOC, "Arrays with a 0 in the shape are not supported");
    hid_t baseType = getH5BaseType<ElemType>();
    hid_t arrayType = H5Tarray_create2(baseType,NDim, dims);
    
    m_writer.createAndStoreDataset(srcGroup.id(),
                                   "data",
                                   arrayType, arrayType,
                                   ndarrayPtr->data());
  }

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

  class EventKeyNotFound : public std::exception {
  public:
  EventKeyNotFound(const ErrSvc::Context &ctx, const PSEvt::EventKey &eventKey, DataTypeLoc dataTypeLoc) 
    : std::exception()
    {
      std::ostringstream str;
      str << "EventKey " << eventKey << " not found in location: " << dataTypeLoc << " [ " << ctx << " ]";
      m_fullMessage = str.str();
    }
    virtual const char * what() const throw() { return m_fullMessage.c_str(); };
    ~EventKeyNotFound() throw() {};
  private:
    std::string m_fullMessage;
  }; // class EventKeyNotFound

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

template <class ElemType, unsigned NDim, bool vlen>
  const std::string Translator::HdfWriterNDArray<ElemType, NDim, vlen>::errorMsgForUnsupportedStride("HdfWriterNDArray::make_datasets, only C or Fortran strides supported");


