//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrayConvert
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/NDArrayConverter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Type.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/DataSet.h"
#include "hdf5pp/Exceptions.h"
#include "hdf5pp/TypeTraits.h"
#include "MsgLogger/MsgLogger.h"
#include "ndarray/ndarray.h"
#include "PSEvt/Proxy.h"
#include "PSEvt/TypeInfoUtils.h"

using namespace std;
using namespace hdf5pp;

namespace {

  const char * logger = "NDArrayConverter";

  // returns one of the NDArrayConverter elem types (int, uint, float, unknown)
  psddl_hdf2psana::NDArrayParameters::ElemType getElemType(hid_t typeId) {
    hid_t typeClass = H5Tget_class(typeId);
    if (typeClass == H5T_INTEGER) {
      H5T_sign_t sign = H5Tget_sign(typeId);
      if (sign == H5T_SGN_2) return psddl_hdf2psana::NDArrayParameters::intElemType;
      if (sign == H5T_SGN_NONE) return psddl_hdf2psana::NDArrayParameters::uintElemType;
      throw Hdf5CallException(ERR_LOC, "H5Tget_sign");
    }
    if (typeClass == H5T_FLOAT) {
      return psddl_hdf2psana::NDArrayParameters::floatElemType;
    }
    return psddl_hdf2psana::NDArrayParameters::unknownElemType;
  }

  // returns information about the type by exploring the hdf5 typeId.
  // If getArrayTypeInfo returns true, then the remaining args will be set
  // as follows:
  // output args:
  //    vlen              - true if typeId is vlen
  //    dims              - if not vlen, the complete dimensions for the array type
  //                        if vlen, the size of dims will be the array rank but 
  //                         dims[0] will be 0 as this is variable and not known until data is read
  //    elemType          - the elemType, int, uint, float or other
  //    sizeBytes         - size in bytes of element
  //
  // If getArrayTypeInfo returns false, the value of the output args is undefined.
  bool getArrayTypeInfo(const hid_t typeId, const hdf5pp::Group &group,
                        bool &vlen, vector<hsize_t> &dims, 
                        psddl_hdf2psana::NDArrayParameters::ElemType &elemType,
                        unsigned &sizeBytes) {    
    H5T_class_t typeClass = H5Tget_class(typeId);
    if (typeClass == H5T_NO_CLASS) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_class on typeId");
    if ( not ((typeClass == H5T_ARRAY) or (typeClass == H5T_VLEN))) {
      MsgLog(logger, error, "typeId not array or vlen in: " << group.name());
      return false;
    }
    vlen = (typeClass == H5T_VLEN);
    if ((not vlen) and (typeClass != H5T_ARRAY)) {
      MsgLog(logger, error, "type is neither vlen nor array");
      return false;
    }
    bool success = false;
    hid_t superType = H5Tget_super(typeId);    // need to close before returning
    if (superType < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_super");
    H5T_class_t superClass = H5Tget_class(superType);
    if (superClass == H5T_NO_CLASS) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_class on super");
    if (vlen) {
      if ((superClass == H5T_INTEGER) or (superClass == H5T_FLOAT)) {
        // 1D array of vlen, don't know slow dim
        elemType = getElemType(superType);
        sizeBytes = H5Tget_size(superType);
        if (sizeBytes == 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_size");
        dims.resize(1);
        dims.at(0)=0;
        success = true;
      } else if (superClass == H5T_ARRAY) {
        // vlen array, 2D or more
        int superRank = H5Tget_array_ndims(superType);
        if (superRank < 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_array_ndims");
        dims.resize(superRank + 1);
        int retVal = H5Tget_array_dims2(superType, &(dims[1]));
        if (retVal != superRank) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_array_dims2");
        dims.at(0)=0;
        hid_t h5elemType = H5Tget_super(superType);
        if (h5elemType < 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_super(super) for vlen");
        elemType = getElemType(h5elemType);
        sizeBytes = H5Tget_size(h5elemType);
        if (sizeBytes == 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_size");
        success = true;
        if (H5Tclose(h5elemType)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose");
      } else {
        MsgLog(logger,error,"vlen but super class is neither array, float, or integer");
      }
    } else {
      // not vlen
      if ((superClass == H5T_INTEGER) or (superClass == H5T_FLOAT)) {
        int rank = H5Tget_array_ndims(typeId);
        if (rank < 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_array_ndims");
        dims.resize(rank);
        int retVal = H5Tget_array_dims2(typeId, &(dims[0]));
        if (retVal != rank) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_array_dims2");
        hid_t h5elemType = superType;
        elemType = getElemType(h5elemType);
        sizeBytes = H5Tget_size(h5elemType);
        if (sizeBytes == 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tget_size");
        success = true;
      } else {
        MsgLog(logger,error,"getArrayTypeInfo, not vlen, but superClass is not integer or float");
      } 
    }
    if (H5Tclose(superType)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose");
    return success;
  }
  

  // define RemoveConst<T>::type to be T without const qualifier
  template <class T>
  struct RemoveConst
  {
    typedef T type;
  };
  template <class T>
  struct RemoveConst<const T>
  {
      typedef T type;
  };


  template<class T>
  class ArrayDelete {
  public:
    void operator()(T*p) { delete[] p; };
  };

  template <class T, unsigned DIM>
  class ProxyNDArray : public PSEvt::Proxy< ndarray<T,DIM> > {
  public:
    ProxyNDArray(const std::vector<hsize_t> &dims, 
                 const hdf5pp::DataSet &ds, 
                 int64_t idx, 
                 bool vlen) 
      : m_dims(dims), m_shape(dims.size()), m_ds(ds), m_idx(idx), m_vlen(vlen)
    {
      for (unsigned idx = 0; idx < dims.size(); ++idx) m_shape[idx]=dims[idx];
    }
  protected:
    typedef typename RemoveConst<T>::type NonConstT;
    virtual boost::shared_ptr<ndarray<T,DIM> > 
    getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
    {
      if (not m_array) {
        readNDArray();
      }
      return m_array;
    }
    void readNDArray() {      
      hsize_t one = 1; 
      hid_t memSpaceId= H5Screate_simple(one, &one, &one);
      if (memSpaceId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Screate_simple");
      hid_t fileSpaceId = H5Dget_space(m_ds.id());
      if (fileSpaceId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Dget_space");
      hsize_t start = m_idx;
      hsize_t count = 1;
      herr_t err = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, &start, NULL, &count, NULL);
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Sselect_hyperslab");
      hid_t elemTypeId = hdf5pp::TypeTraits<NonConstT>::native_type().id();
      if (elemTypeId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "TypeTraits native type");
      boost::shared_ptr<NonConstT> data;
      unsigned slowDim = 0;
      if (m_vlen) {
        data = readVlenData(elemTypeId, memSpaceId, fileSpaceId, slowDim);
        m_shape[0]=slowDim;
      }
      else {
        data = readNonVlenData(elemTypeId, memSpaceId, fileSpaceId);
      }
      err = H5Sclose(fileSpaceId);
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Sclose on fileSpaceId");
      err = H5Sclose(memSpaceId);
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Sclose on memSpaceId");
      const boost::shared_ptr<T> possiblyConstData = data;
      m_array = boost::make_shared< ndarray<T,DIM> >(possiblyConstData, 
                                                     (const unsigned int *)&(m_shape.at(0)), 
                                                     ndns::C );
    }
    
    boost::shared_ptr<NonConstT> readVlenData(const hid_t elemTypeId, const hid_t memSpaceId, 
                                              const hid_t fileSpaceId, unsigned &slowDim) {
      hvl_t vdata;
      hid_t baseTypeId = -1;
      
      if ( DIM == 1 ) baseTypeId = elemTypeId;
      else baseTypeId = H5Tarray_create2( elemTypeId , DIM-1 , & m_dims[1] );
      if (baseTypeId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tarray_create2 or native type");
      
      hid_t vlenTypeId = H5Tvlen_create( baseTypeId );
      if (vlenTypeId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tvlen_create");
      
      herr_t err = H5Dread( m_ds.id(), vlenTypeId, memSpaceId, fileSpaceId, H5P_DEFAULT, &vdata);
      
      // check read, close memory types
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Dread");

      unsigned long numberOfElements = 1;
      for (unsigned idx = 1; idx < DIM; ++idx) numberOfElements *= m_dims[idx];
      numberOfElements *= vdata.len;
      slowDim = vdata.len;

      boost::shared_ptr<NonConstT> data(new NonConstT[numberOfElements], ArrayDelete<NonConstT>());
      // copy data to shared pointer for ndarray
      NonConstT *src = (NonConstT *)vdata.p;
      NonConstT *dest = data.get();
      for (unsigned idx = 0; idx < numberOfElements; ++idx) *dest++ = *src++;
      
      // free data in vlen structure
      err = H5Dvlen_reclaim(vlenTypeId, memSpaceId, H5P_DEFAULT, &vdata);
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Dvlen_reclaim");

      if (H5Tclose(vlenTypeId)<0) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5TClose on vlenTypeId");
            
      if (DIM > 1) {
        err = H5Tclose(baseTypeId);
        if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose on baseTypeId");
      }
      return data;
    }
    
    boost::shared_ptr<NonConstT>  readNonVlenData(hid_t elemTypeId, hid_t memSpaceId, hid_t fileSpaceId) {
      hid_t memTypeId= -1;
      memTypeId = H5Tarray_create2( elemTypeId , DIM , & m_dims[0] );
      if (memTypeId < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tarray_create2");
      unsigned long numberOfElements = 1;
      for (unsigned idx = 0; idx < DIM; ++idx) numberOfElements *= m_dims[idx];
      boost::shared_ptr<NonConstT> data(new NonConstT[numberOfElements], ArrayDelete<NonConstT>());
      herr_t err = H5Dread(m_ds.id(), memTypeId, memSpaceId, fileSpaceId, H5P_DEFAULT, data.get());
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Dread");
      err = H5Tclose(memTypeId);
      if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose on memTypeId (in readNonVlenData)");
      return data;
    }
    
  private:
    const std::vector<hsize_t> m_dims;
    std::vector<unsigned> m_shape;
    const hdf5pp::DataSet m_ds;
    const int64_t m_idx;
    bool m_vlen;
    boost::shared_ptr<ndarray<T,DIM> > m_array;
  }; // class ProxyNDArray

  template <class ElemType, unsigned rank>
  void NDArrayConvert(const std::vector<hsize_t> &dims, bool vlen, 
                      int schema_version, const hdf5pp::DataSet &ds, 
                      int64_t idx, const Pds::Src &src, const std::string & key, PSEvt::Event &evt) 
  {
    MsgLog(logger,debug, "in NDArrayConvert<"             
           << PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(ElemType))
           << ", " << rank << " > idx=" << idx << " src=" << src << " key= " << key);
    
    boost::shared_ptr< PSEvt::Proxy< ndarray< ElemType, rank> > > proxy;
    switch (schema_version) {
    case 0:
      proxy = boost::make_shared< ProxyNDArray<ElemType, rank> >(dims, ds, idx, vlen);
      evt.putProxy(proxy, src, key);
      break;
    default:
      MsgLog(logger,warning, "NDArrayConvert<"             
             << PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(ElemType))
             << ", " << rank << " > unknown schema: " << schema_version);
    }
  }

  template<unsigned rank>
  void NDArrayConvert(bool vlen, const std::vector<hsize_t> &dims, 
                      psddl_hdf2psana::NDArrayParameters::ElemType elemType,
                      unsigned sizeBytes,
                      int schema_version, const hdf5pp::DataSet &ds, int64_t idx,
                      bool isConst,
                      const Pds::Src &src, const std::string &key, PSEvt::Event &evt)
  {
    if (rank != dims.size()) throw ErrSvc::Issue(ERR_LOC, "NDArrayConvert: rank != dims.size()");
    switch (elemType) {
    case psddl_hdf2psana::NDArrayParameters::intElemType: 
      switch (sizeBytes) {
      case 1:
        if (isConst) NDArrayConvert<const int8_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<int8_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        break;
      case 2:
        if (isConst) NDArrayConvert<const int16_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<int16_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        break;
      case 4:
        if (isConst) NDArrayConvert<const int32_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<int32_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        break;
      case 8:
        if (isConst) NDArrayConvert<const int64_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<int64_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        break;
      default:
        MsgLog(logger,warning,"array elem type has signed integer class with unsupported size of " 
               << sizeBytes << " bytes.");
        return;
      }
      break;
      case psddl_hdf2psana::NDArrayParameters::uintElemType: 
        switch (sizeBytes) {
        case 1:
          if (isConst) NDArrayConvert<const uint8_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          else NDArrayConvert<uint8_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          break;
        case 2:
          if (isConst) NDArrayConvert<const uint16_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          else NDArrayConvert<uint16_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          break;
        case 4:
          if (isConst) NDArrayConvert<const uint32_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          else NDArrayConvert<uint32_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          break;
        case 8:
          if (isConst) NDArrayConvert<const uint64_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          else NDArrayConvert<uint64_t, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
          break;
        default:
          MsgLog(logger,warning,"array elem type has signed integer class with unsupported size of " 
                 << sizeBytes << " bytes.");
          return;
        }
        break;
    case psddl_hdf2psana::NDArrayParameters::floatElemType:
      if (sizeBytes <= 4) {
        if (sizeof(float)<4) MsgLog(logger, warning, "sizeof(float)<4");
        if (isConst) NDArrayConvert<const float, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<float, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
      } else if (sizeBytes <= 8) {
        if (sizeof(double)<8) MsgLog(logger, warning, "sizeof(double)<8");
        if (isConst) NDArrayConvert<const double, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
        else NDArrayConvert<double, rank>(dims, vlen, schema_version, ds, idx, src, key, evt);
      }
      break;
    case psddl_hdf2psana::NDArrayParameters::unknownElemType:
      MsgLog(logger,warning,"unsupported elem type, no conversion");
      return;
    }
  } // NDArrayConvert<rank>

}; // local namespace


namespace psddl_hdf2psana {

void  NDArrayConverter::convert(const hdf5pp::Group& group, int64_t idx, 
                                const NDArrayParameters &ndArrayParams,
                                int schema_version, 
                                const Pds::Src &src, const std::string & key, 
                                PSEvt::Event &evt) const
{
  hdf5pp::DataSet ds = group.openDataSet("data");
  hdf5pp::Type type = ds.type();

  bool vlen;
  vector<hsize_t> dims;
  psddl_hdf2psana::NDArrayParameters::ElemType elemType;
  unsigned sizeBytes;  

  bool success = getArrayTypeInfo(type.id(), group, vlen, dims,
                                  elemType, sizeBytes);

  if ((not success) or (elemType == psddl_hdf2psana::NDArrayParameters::unknownElemType)) {
    MsgLog(logger, error, "dataset type information incorrect for ndarray convert. "
           << " getArrayTypeInfo success=" << success << " unknownElemType="
           << (elemType == psddl_hdf2psana::NDArrayParameters::unknownElemType));
    return;
  }
  if (vlen != ndArrayParams.isVlen()) {
    MsgLog(logger, warning, "vlen from group attribute disagrees with hdf5 type: group=" << group.name());
  }
  if (elemType != ndArrayParams.elemType()) {
    MsgLog(logger, warning, "elemType from group attribute disagrees with hdf5 type: group=" << group.name());
  }
  if (sizeBytes != ndArrayParams.sizeBytes()) {
    MsgLog(logger, warning, "sizeBytes from group attribute disagrees with hdf5 type: group=" << group.name());
  }
  unsigned rank = dims.size();
  if ( rank != ndArrayParams.dim()) {
    MsgLog(logger, warning, "rank from group attributes disagrees with hdf5 type: group=" << group.name());
  }
  switch (rank) {
  case 1:
    ::NDArrayConvert<1>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  case 2:
    ::NDArrayConvert<2>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  case 3:
    ::NDArrayConvert<3>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  case 4:
    ::NDArrayConvert<4>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  case 5:
    ::NDArrayConvert<5>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  case 6:
    ::NDArrayConvert<6>(vlen, dims, elemType, sizeBytes, 
                        schema_version, ds, idx, ndArrayParams.isConstElem(), src, key, evt);
    break;
  default:
    MsgLog(logger, error, "rank = " << rank
           << " is outside range [1,6] for NDArray conversion - group = " << group.name()
           << " idx=" << idx);
    return;
  }
}

} // namespace psddl_hdf2psana
