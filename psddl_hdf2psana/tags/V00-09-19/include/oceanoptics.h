#ifndef PSDDL_HDF2PSANA_OCEANOPTICS_H
#define PSDDL_HDF2PSANA_OCEANOPTICS_H 1


#include "psddl_psana/oceanoptics.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "psddl_hdf2psana/oceanoptics.ddl.h"


/*
 * This specialization for oceanoptics DataV* classes is needed because
 * we want to save special dataset "corrSpectra" which contains corrected
 * values in spectra array. DDL does not provide separate method for
 * accessing corrected array so we cannot write HDF5 DDL and have to write
 * C++ code for it. (It may be possible to modify DDL to return corrected
 * array but it may cause other complications).
 */

namespace psddl_hdf2psana {
namespace OceanOptics {

namespace ns_DataV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::OceanOptics::DataV1& psanaobj);
  ~dataset_data();

  uint64_t frameCounter;
  uint64_t numDelayedFrames;
  uint64_t numDiscardFrames;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameStart;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameFirstData;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameEnd;
  int8_t numSpectraInData;
  int8_t numSpectraInQueue;
  int8_t numSpectraUnused;
  double durationOfFrame;


};
}


template <typename Config>
class DataV1_v0 : public Psana::OceanOptics::DataV1 {
public:
  typedef Psana::OceanOptics::DataV1 PsanaType;
  DataV1_v0() {}
  DataV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV1_v0() {}
  virtual ndarray<const uint16_t, 1> data() const;
  virtual uint64_t frameCounter() const;
  virtual uint64_t numDelayedFrames() const;
  virtual uint64_t numDiscardFrames() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameStart() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameFirstData() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameEnd() const;
  virtual int8_t numSpectraInData() const;
  virtual int8_t numSpectraInQueue() const;
  virtual int8_t numSpectraUnused() const;
  virtual double durationOfFrame() const;
    double nonlinerCorrected(uint32_t iPixel) const;

private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;
  mutable ndarray<const uint16_t, 1> m_ds_spectra;
  void read_ds_spectra() const;
  mutable boost::shared_ptr<OceanOptics::ns_DataV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameStart;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameFirstData;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameEnd;
};

void make_datasets_DataV1_v0(const Psana::OceanOptics::DataV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV1_v0(const Psana::OceanOptics::DataV1* obj, hdf5pp::Group group, long index, bool append);





namespace ns_DataV2_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::OceanOptics::DataV2& psanaobj);
  ~dataset_data();

  uint64_t frameCounter;
  uint64_t numDelayedFrames;
  uint64_t numDiscardFrames;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameStart;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameFirstData;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameEnd;
  int8_t numSpectraInData;
  int8_t numSpectraInQueue;
  int8_t numSpectraUnused;
  double durationOfFrame;


};
}


template <typename Config>
class DataV2_v0 : public Psana::OceanOptics::DataV2 {
public:
  typedef Psana::OceanOptics::DataV2 PsanaType;
  DataV2_v0() {}
  DataV2_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV2_v0() {}
  virtual ndarray<const uint16_t, 1> data() const;
  virtual uint64_t frameCounter() const;
  virtual uint64_t numDelayedFrames() const;
  virtual uint64_t numDiscardFrames() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameStart() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameFirstData() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameEnd() const;
  virtual int8_t numSpectraInData() const;
  virtual int8_t numSpectraInQueue() const;
  virtual int8_t numSpectraUnused() const;
  virtual double durationOfFrame() const;
    double nonlinerCorrected(uint32_t iPixel) const;

private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;
  mutable ndarray<const uint16_t, 1> m_ds_spectra;
  void read_ds_spectra() const;
  mutable boost::shared_ptr<OceanOptics::ns_DataV2_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameStart;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameFirstData;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameEnd;
};

void make_datasets_DataV2_v0(const Psana::OceanOptics::DataV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV2_v0(const Psana::OceanOptics::DataV2* obj, hdf5pp::Group group, long index, bool append);

// DataV3
namespace ns_DataV3_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::OceanOptics::DataV3& psanaobj);
  ~dataset_data();

  uint64_t frameCounter;
  uint64_t numDelayedFrames;
  uint64_t numDiscardFrames;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameStart;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameFirstData;
  OceanOptics::ns_timespec64_v0::dataset_data timeFrameEnd;
  int8_t numSpectraInData;
  int8_t numSpectraInQueue;
  int8_t numSpectraUnused;
  double durationOfFrame;


};
}


template <typename Config>
class DataV3_v0 : public Psana::OceanOptics::DataV3 {
public:
  typedef Psana::OceanOptics::DataV3 PsanaType;
  DataV3_v0() {}
  DataV3_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV3_v0() {}
  virtual ndarray<const uint16_t, 1> data() const;
  virtual uint64_t frameCounter() const;
  virtual uint64_t numDelayedFrames() const;
  virtual uint64_t numDiscardFrames() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameStart() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameFirstData() const;
  virtual const Psana::OceanOptics::timespec64& timeFrameEnd() const;
  virtual int8_t numSpectraInData() const;
  virtual int8_t numSpectraInQueue() const;
  virtual int8_t numSpectraUnused() const;
  virtual double durationOfFrame() const;
    double nonlinerCorrected(uint32_t iPixel) const;

private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;
  mutable ndarray<const uint16_t, 1> m_ds_spectra;
  void read_ds_spectra() const;
  mutable boost::shared_ptr<OceanOptics::ns_DataV3_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameStart;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameFirstData;
  mutable Psana::OceanOptics::timespec64 m_ds_storage_data_timeFrameEnd;
};

void make_datasets_DataV3_v0(const Psana::OceanOptics::DataV3& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV3_v0(const Psana::OceanOptics::DataV3* obj, hdf5pp::Group group, long index, bool append);



} // namespace OceanOptics
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_OCEANOPTICS_DDL_H
