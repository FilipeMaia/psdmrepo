
// *** Do not edit this file, it is auto-generated ***

#include "psddl_hdf2psana/camera.ddl.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/Utils.h"
#include "PSEvt/DataProxy.h"
#include "psddl_hdf2psana/Exceptions.h"
#include "psddl_hdf2psana/camera.h"
namespace psddl_hdf2psana {
namespace Camera {

hdf5pp::Type ns_FrameCoord_v0_dataset_data_stored_type()
{
  typedef ns_FrameCoord_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("column", offsetof(DsType, column), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("row", offsetof(DsType, row), hdf5pp::TypeTraits<uint16_t>::stored_type());
  return type;
}

hdf5pp::Type ns_FrameCoord_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_FrameCoord_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_FrameCoord_v0_dataset_data_native_type()
{
  typedef ns_FrameCoord_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("column", offsetof(DsType, column), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("row", offsetof(DsType, row), hdf5pp::TypeTraits<uint16_t>::native_type());
  return type;
}

hdf5pp::Type ns_FrameCoord_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_FrameCoord_v0_dataset_data_native_type();
  return type;
}

ns_FrameCoord_v0::dataset_data::dataset_data()
{
}

ns_FrameCoord_v0::dataset_data::dataset_data(const Psana::Camera::FrameCoord& psanaobj)
  : column(psanaobj.column())
  , row(psanaobj.row())
{
}

ns_FrameCoord_v0::dataset_data::~dataset_data()
{
}
boost::shared_ptr<Psana::Camera::FrameCoord>
Proxy_FrameCoord_v0::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (not m_data) {
    boost::shared_ptr<Camera::ns_FrameCoord_v0::dataset_data> ds_data = hdf5pp::Utils::readGroup<Camera::ns_FrameCoord_v0::dataset_data>(m_group, "data", m_idx);
    m_data.reset(new PsanaType(ds_data->column, ds_data->row));
  }
  return m_data;
}


void make_datasets_FrameCoord_v0(const Psana::Camera::FrameCoord& obj, 
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = Camera::ns_FrameCoord_v0::dataset_data::stored_type();
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);    
  }
}

void store_FrameCoord_v0(const Psana::Camera::FrameCoord* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    Camera::ns_FrameCoord_v0::dataset_data ds_data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "data", ds_data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "data", ds_data);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
  }
}

boost::shared_ptr<PSEvt::Proxy<Psana::Camera::FrameCoord> > make_FrameCoord(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<Proxy_FrameCoord_v0>(group, idx);
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameCoord> >(boost::shared_ptr<Psana::Camera::FrameCoord>());
  }
}

void make_datasets(const Psana::Camera::FrameCoord& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version)
{
  if (version < 0) version = 0;
  group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    make_datasets_FrameCoord_v0(obj, group, chunkPolicy, deflate, shuffle);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameCoord", version);
  }
}

void store_FrameCoord(const Psana::Camera::FrameCoord* obj, hdf5pp::Group group, long index, int version, bool append)
{
  if (version < 0) version = 0;
  if (not append) group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    store_FrameCoord_v0(obj, group, index, append);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameCoord", version);
  }
}

void store(const Psana::Camera::FrameCoord& obj, hdf5pp::Group group, int version) 
{
  store_FrameCoord(&obj, group, 0, version, false);
}

void store_at(const Psana::Camera::FrameCoord* obj, hdf5pp::Group group, long index, int version)
{
  store_FrameCoord(obj, group, index, version, true);
}


void make_datasets_FrameFccdConfigV1_v0(const Psana::Camera::FrameFccdConfigV1& obj, 
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
}

void store_FrameFccdConfigV1_v0(const Psana::Camera::FrameFccdConfigV1* obj, hdf5pp::Group group, long index, bool append)
{
}

boost::shared_ptr<PSEvt::Proxy<Psana::Camera::FrameFccdConfigV1> > make_FrameFccdConfigV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameFccdConfigV1> >(boost::make_shared<FrameFccdConfigV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameFccdConfigV1> >(boost::shared_ptr<Psana::Camera::FrameFccdConfigV1>());
  }
}

void make_datasets(const Psana::Camera::FrameFccdConfigV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version)
{
  if (version < 0) version = 0;
  group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    make_datasets_FrameFccdConfigV1_v0(obj, group, chunkPolicy, deflate, shuffle);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameFccdConfigV1", version);
  }
}

void store_FrameFccdConfigV1(const Psana::Camera::FrameFccdConfigV1* obj, hdf5pp::Group group, long index, int version, bool append)
{
  if (version < 0) version = 0;
  if (not append) group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    store_FrameFccdConfigV1_v0(obj, group, index, append);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameFccdConfigV1", version);
  }
}

void store(const Psana::Camera::FrameFccdConfigV1& obj, hdf5pp::Group group, int version) 
{
  store_FrameFccdConfigV1(&obj, group, 0, version, false);
}

void store_at(const Psana::Camera::FrameFccdConfigV1* obj, hdf5pp::Group group, long index, int version)
{
  store_FrameFccdConfigV1(obj, group, index, version, true);
}


hdf5pp::Type ns_FrameFexConfigV1_v0_dataset_config_stored_type()
{
  typedef ns_FrameFexConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::EnumType<uint32_t> _enum_type_forwarding = hdf5pp::EnumType<uint32_t>::enumType();
  _enum_type_forwarding.insert("NoFrame", Psana::Camera::FrameFexConfigV1::NoFrame);
  _enum_type_forwarding.insert("FullFrame", Psana::Camera::FrameFexConfigV1::FullFrame);
  _enum_type_forwarding.insert("RegionOfInterest", Psana::Camera::FrameFexConfigV1::RegionOfInterest);
  type.insert("forwarding", offsetof(DsType, forwarding), _enum_type_forwarding);
  type.insert("forward_prescale", offsetof(DsType, forward_prescale), hdf5pp::TypeTraits<uint32_t>::stored_type());
  hdf5pp::EnumType<uint32_t> _enum_type_processing = hdf5pp::EnumType<uint32_t>::enumType();
  _enum_type_processing.insert("NoProcessing", Psana::Camera::FrameFexConfigV1::NoProcessing);
  _enum_type_processing.insert("GssFullFrame", Psana::Camera::FrameFexConfigV1::GssFullFrame);
  _enum_type_processing.insert("GssRegionOfInterest", Psana::Camera::FrameFexConfigV1::GssRegionOfInterest);
  _enum_type_processing.insert("GssThreshold", Psana::Camera::FrameFexConfigV1::GssThreshold);
  type.insert("processing", offsetof(DsType, processing), _enum_type_processing);
  type.insert("roiBegin", offsetof(DsType, roiBegin), hdf5pp::TypeTraits<Camera::ns_FrameCoord_v0::dataset_data>::stored_type());
  type.insert("roiEnd", offsetof(DsType, roiEnd), hdf5pp::TypeTraits<Camera::ns_FrameCoord_v0::dataset_data>::stored_type());
  type.insert("threshold", offsetof(DsType, threshold), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("number_of_masked_pixels", offsetof(DsType, number_of_masked_pixels), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_FrameFexConfigV1_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_FrameFexConfigV1_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_FrameFexConfigV1_v0_dataset_config_native_type()
{
  typedef ns_FrameFexConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::EnumType<uint32_t> _enum_type_forwarding = hdf5pp::EnumType<uint32_t>::enumType();
  _enum_type_forwarding.insert("NoFrame", Psana::Camera::FrameFexConfigV1::NoFrame);
  _enum_type_forwarding.insert("FullFrame", Psana::Camera::FrameFexConfigV1::FullFrame);
  _enum_type_forwarding.insert("RegionOfInterest", Psana::Camera::FrameFexConfigV1::RegionOfInterest);
  type.insert("forwarding", offsetof(DsType, forwarding), _enum_type_forwarding);
  type.insert("forward_prescale", offsetof(DsType, forward_prescale), hdf5pp::TypeTraits<uint32_t>::native_type());
  hdf5pp::EnumType<uint32_t> _enum_type_processing = hdf5pp::EnumType<uint32_t>::enumType();
  _enum_type_processing.insert("NoProcessing", Psana::Camera::FrameFexConfigV1::NoProcessing);
  _enum_type_processing.insert("GssFullFrame", Psana::Camera::FrameFexConfigV1::GssFullFrame);
  _enum_type_processing.insert("GssRegionOfInterest", Psana::Camera::FrameFexConfigV1::GssRegionOfInterest);
  _enum_type_processing.insert("GssThreshold", Psana::Camera::FrameFexConfigV1::GssThreshold);
  type.insert("processing", offsetof(DsType, processing), _enum_type_processing);
  type.insert("roiBegin", offsetof(DsType, roiBegin), hdf5pp::TypeTraits<Camera::ns_FrameCoord_v0::dataset_data>::native_type());
  type.insert("roiEnd", offsetof(DsType, roiEnd), hdf5pp::TypeTraits<Camera::ns_FrameCoord_v0::dataset_data>::native_type());
  type.insert("threshold", offsetof(DsType, threshold), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("number_of_masked_pixels", offsetof(DsType, number_of_masked_pixels), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_FrameFexConfigV1_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_FrameFexConfigV1_v0_dataset_config_native_type();
  return type;
}

ns_FrameFexConfigV1_v0::dataset_config::dataset_config()
{
}

ns_FrameFexConfigV1_v0::dataset_config::dataset_config(const Psana::Camera::FrameFexConfigV1& psanaobj)
  : forwarding(psanaobj.forwarding())
  , forward_prescale(psanaobj.forward_prescale())
  , processing(psanaobj.processing())
  , roiBegin(psanaobj.roiBegin())
  , roiEnd(psanaobj.roiEnd())
  , threshold(psanaobj.threshold())
  , number_of_masked_pixels(psanaobj.number_of_masked_pixels())
{
}

ns_FrameFexConfigV1_v0::dataset_config::~dataset_config()
{
}
Psana::Camera::FrameFexConfigV1::Forwarding FrameFexConfigV1_v0::forwarding() const {
  if (not m_ds_config) read_ds_config();
  return Psana::Camera::FrameFexConfigV1::Forwarding(m_ds_config->forwarding);
}
uint32_t FrameFexConfigV1_v0::forward_prescale() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->forward_prescale);
}
Psana::Camera::FrameFexConfigV1::Processing FrameFexConfigV1_v0::processing() const {
  if (not m_ds_config) read_ds_config();
  return Psana::Camera::FrameFexConfigV1::Processing(m_ds_config->processing);
}
const Psana::Camera::FrameCoord& FrameFexConfigV1_v0::roiBegin() const {
  if (not m_ds_config) read_ds_config();
  m_ds_storage_config_roiBegin = Psana::Camera::FrameCoord(m_ds_config->roiBegin);
  return m_ds_storage_config_roiBegin;
}
const Psana::Camera::FrameCoord& FrameFexConfigV1_v0::roiEnd() const {
  if (not m_ds_config) read_ds_config();
  m_ds_storage_config_roiEnd = Psana::Camera::FrameCoord(m_ds_config->roiEnd);
  return m_ds_storage_config_roiEnd;
}
uint32_t FrameFexConfigV1_v0::threshold() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->threshold);
}
uint32_t FrameFexConfigV1_v0::number_of_masked_pixels() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->number_of_masked_pixels);
}
ndarray<const Psana::Camera::FrameCoord, 1> FrameFexConfigV1_v0::masked_pixel_coordinates() const {
  if (m_ds_masked_pixel_coordinates.empty()) read_ds_masked_pixel_coordinates();
  return m_ds_masked_pixel_coordinates;
}
void FrameFexConfigV1_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<Camera::ns_FrameFexConfigV1_v0::dataset_config>(m_group, "config", m_idx);
}
void FrameFexConfigV1_v0::read_ds_masked_pixel_coordinates() const {
  ndarray<Camera::ns_FrameCoord_v0::dataset_data, 1> arr = hdf5pp::Utils::readNdarray<Camera::ns_FrameCoord_v0::dataset_data, 1>(m_group, "masked_pixel_coordinates", m_idx);
  ndarray<Psana::Camera::FrameCoord, 1> tmp(arr.shape());
  std::copy(arr.begin(), arr.end(), tmp.begin());
  m_ds_masked_pixel_coordinates = tmp;
}

void make_datasets_FrameFexConfigV1_v0(const Psana::Camera::FrameFexConfigV1& obj, 
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = Camera::ns_FrameFexConfigV1_v0::dataset_config::stored_type();
    hdf5pp::Utils::createDataset(group, "config", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);    
  }
  {
    typedef __typeof__(obj.masked_pixel_coordinates()) PsanaArray;
    const PsanaArray& psana_array = obj.masked_pixel_coordinates();
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<Camera::ns_FrameCoord_v0::dataset_data>::stored_type(), psana_array.shape()[0]);
    hdf5pp::Utils::createDataset(group, "masked_pixel_coordinates", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);    
  }
}

void store_FrameFexConfigV1_v0(const Psana::Camera::FrameFexConfigV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    Camera::ns_FrameFexConfigV1_v0::dataset_config ds_data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "config", ds_data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "config", ds_data);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "config", index < 0 ? index : index + 1);
  }
  if (obj) {
    typedef __typeof__(obj->masked_pixel_coordinates()) PsanaArray;
    typedef ndarray<Camera::ns_FrameCoord_v0::dataset_data, 1> HdfArray;
    PsanaArray psana_array = obj->masked_pixel_coordinates();
    HdfArray hdf_array(psana_array.shape());
    HdfArray::iterator out = hdf_array.begin();
    for (PsanaArray::iterator it = psana_array.begin(); it != psana_array.end(); ++ it, ++ out) {
      *out = Camera::ns_FrameCoord_v0::dataset_data(*it);
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "masked_pixel_coordinates", hdf_array, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "masked_pixel_coordinates", hdf_array);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "masked_pixel_coordinates", index < 0 ? index : index + 1);
  }
}

boost::shared_ptr<PSEvt::Proxy<Psana::Camera::FrameFexConfigV1> > make_FrameFexConfigV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameFexConfigV1> >(boost::make_shared<FrameFexConfigV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameFexConfigV1> >(boost::shared_ptr<Psana::Camera::FrameFexConfigV1>());
  }
}

void make_datasets(const Psana::Camera::FrameFexConfigV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version)
{
  if (version < 0) version = 0;
  group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    make_datasets_FrameFexConfigV1_v0(obj, group, chunkPolicy, deflate, shuffle);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameFexConfigV1", version);
  }
}

void store_FrameFexConfigV1(const Psana::Camera::FrameFexConfigV1* obj, hdf5pp::Group group, long index, int version, bool append)
{
  if (version < 0) version = 0;
  if (not append) group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    store_FrameFexConfigV1_v0(obj, group, index, append);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameFexConfigV1", version);
  }
}

void store(const Psana::Camera::FrameFexConfigV1& obj, hdf5pp::Group group, int version) 
{
  store_FrameFexConfigV1(&obj, group, 0, version, false);
}

void store_at(const Psana::Camera::FrameFexConfigV1* obj, hdf5pp::Group group, long index, int version)
{
  store_FrameFexConfigV1(obj, group, index, version, true);
}

boost::shared_ptr<PSEvt::Proxy<Psana::Camera::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameV1> >(boost::make_shared<FrameV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::FrameV1> >(boost::shared_ptr<Psana::Camera::FrameV1>());
  }
}

void make_datasets(const Psana::Camera::FrameV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version)
{
  if (version < 0) version = 0;
  group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    make_datasets_FrameV1_v0(obj, group, chunkPolicy, deflate, shuffle);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameV1", version);
  }
}

void store_FrameV1(const Psana::Camera::FrameV1* obj, hdf5pp::Group group, long index, int version, bool append)
{
  if (version < 0) version = 0;
  if (not append) group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    store_FrameV1_v0(obj, group, index, append);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.FrameV1", version);
  }
}

void store(const Psana::Camera::FrameV1& obj, hdf5pp::Group group, int version) 
{
  store_FrameV1(&obj, group, 0, version, false);
}

void store_at(const Psana::Camera::FrameV1* obj, hdf5pp::Group group, long index, int version)
{
  store_FrameV1(obj, group, index, version, true);
}


hdf5pp::Type ns_TwoDGaussianV1_v0_dataset_data_stored_type()
{
  typedef ns_TwoDGaussianV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("integral", offsetof(DsType, integral), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("xmean", offsetof(DsType, xmean), hdf5pp::TypeTraits<double>::stored_type());
  type.insert("ymean", offsetof(DsType, ymean), hdf5pp::TypeTraits<double>::stored_type());
  type.insert("major_axis_width", offsetof(DsType, major_axis_width), hdf5pp::TypeTraits<double>::stored_type());
  type.insert("minor_axis_width", offsetof(DsType, minor_axis_width), hdf5pp::TypeTraits<double>::stored_type());
  type.insert("major_axis_tilt", offsetof(DsType, major_axis_tilt), hdf5pp::TypeTraits<double>::stored_type());
  return type;
}

hdf5pp::Type ns_TwoDGaussianV1_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_TwoDGaussianV1_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_TwoDGaussianV1_v0_dataset_data_native_type()
{
  typedef ns_TwoDGaussianV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("integral", offsetof(DsType, integral), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("xmean", offsetof(DsType, xmean), hdf5pp::TypeTraits<double>::native_type());
  type.insert("ymean", offsetof(DsType, ymean), hdf5pp::TypeTraits<double>::native_type());
  type.insert("major_axis_width", offsetof(DsType, major_axis_width), hdf5pp::TypeTraits<double>::native_type());
  type.insert("minor_axis_width", offsetof(DsType, minor_axis_width), hdf5pp::TypeTraits<double>::native_type());
  type.insert("major_axis_tilt", offsetof(DsType, major_axis_tilt), hdf5pp::TypeTraits<double>::native_type());
  return type;
}

hdf5pp::Type ns_TwoDGaussianV1_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_TwoDGaussianV1_v0_dataset_data_native_type();
  return type;
}

ns_TwoDGaussianV1_v0::dataset_data::dataset_data()
{
}

ns_TwoDGaussianV1_v0::dataset_data::dataset_data(const Psana::Camera::TwoDGaussianV1& psanaobj)
  : integral(psanaobj.integral())
  , xmean(psanaobj.xmean())
  , ymean(psanaobj.ymean())
  , major_axis_width(psanaobj.major_axis_width())
  , minor_axis_width(psanaobj.minor_axis_width())
  , major_axis_tilt(psanaobj.major_axis_tilt())
{
}

ns_TwoDGaussianV1_v0::dataset_data::~dataset_data()
{
}
uint64_t TwoDGaussianV1_v0::integral() const {
  if (not m_ds_data) read_ds_data();
  return uint64_t(m_ds_data->integral);
}
double TwoDGaussianV1_v0::xmean() const {
  if (not m_ds_data) read_ds_data();
  return double(m_ds_data->xmean);
}
double TwoDGaussianV1_v0::ymean() const {
  if (not m_ds_data) read_ds_data();
  return double(m_ds_data->ymean);
}
double TwoDGaussianV1_v0::major_axis_width() const {
  if (not m_ds_data) read_ds_data();
  return double(m_ds_data->major_axis_width);
}
double TwoDGaussianV1_v0::minor_axis_width() const {
  if (not m_ds_data) read_ds_data();
  return double(m_ds_data->minor_axis_width);
}
double TwoDGaussianV1_v0::major_axis_tilt() const {
  if (not m_ds_data) read_ds_data();
  return double(m_ds_data->major_axis_tilt);
}
void TwoDGaussianV1_v0::read_ds_data() const {
  m_ds_data = hdf5pp::Utils::readGroup<Camera::ns_TwoDGaussianV1_v0::dataset_data>(m_group, "data", m_idx);
}

void make_datasets_TwoDGaussianV1_v0(const Psana::Camera::TwoDGaussianV1& obj, 
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = Camera::ns_TwoDGaussianV1_v0::dataset_data::stored_type();
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);    
  }
}

void store_TwoDGaussianV1_v0(const Psana::Camera::TwoDGaussianV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    Camera::ns_TwoDGaussianV1_v0::dataset_data ds_data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "data", ds_data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "data", ds_data);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
  }
}

boost::shared_ptr<PSEvt::Proxy<Psana::Camera::TwoDGaussianV1> > make_TwoDGaussianV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::TwoDGaussianV1> >(boost::make_shared<TwoDGaussianV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Camera::TwoDGaussianV1> >(boost::shared_ptr<Psana::Camera::TwoDGaussianV1>());
  }
}

void make_datasets(const Psana::Camera::TwoDGaussianV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version)
{
  if (version < 0) version = 0;
  group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    make_datasets_TwoDGaussianV1_v0(obj, group, chunkPolicy, deflate, shuffle);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.TwoDGaussianV1", version);
  }
}

void store_TwoDGaussianV1(const Psana::Camera::TwoDGaussianV1* obj, hdf5pp::Group group, long index, int version, bool append)
{
  if (version < 0) version = 0;
  if (not append) group.createAttr<uint32_t>("_schemaVersion").store(version);
  switch (version) {
  case 0:
    store_TwoDGaussianV1_v0(obj, group, index, append);
    break;
  default:
    throw ExceptionSchemaVersion(ERR_LOC, "Camera.TwoDGaussianV1", version);
  }
}

void store(const Psana::Camera::TwoDGaussianV1& obj, hdf5pp::Group group, int version) 
{
  store_TwoDGaussianV1(&obj, group, 0, version, false);
}

void store_at(const Psana::Camera::TwoDGaussianV1* obj, hdf5pp::Group group, long index, int version)
{
  store_TwoDGaussianV1(obj, group, index, version, true);
}

} // namespace Camera
} // namespace psddl_hdf2psana
