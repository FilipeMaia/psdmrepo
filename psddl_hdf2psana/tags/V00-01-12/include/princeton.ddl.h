#ifndef PSDDL_HDF2PSANA_PRINCETON_DDL_H
#define PSDDL_HDF2PSANA_PRINCETON_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "psddl_psana/princeton.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
namespace psddl_hdf2psana {
namespace Princeton {

namespace ns_ConfigV1_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint32_t width; 
  uint32_t height; 
  uint32_t orgX; 
  uint32_t orgY; 
  uint32_t binX; 
  uint32_t binY; 
  float exposureTime; 
  float coolingTemp; 
  uint32_t readoutSpeedIndex; 
  uint16_t readoutEventCode; 
  uint16_t delayMode; 
  uint32_t frameSize; 
  uint32_t numPixelsX; 
  uint32_t numPixelsY; 
  uint32_t numPixels; 

};
}


class ConfigV1_v0 : public Psana::Princeton::ConfigV1 {
public:
  typedef Psana::Princeton::ConfigV1 PsanaType;
  ConfigV1_v0() {}
  ConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  ConfigV1_v0(const boost::shared_ptr<Princeton::ns_ConfigV1_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~ConfigV1_v0() {}
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t orgX() const;
  virtual uint32_t orgY() const;
  virtual uint32_t binX() const;
  virtual uint32_t binY() const;
  virtual float exposureTime() const;
  virtual float coolingTemp() const;
  virtual uint32_t readoutSpeedIndex() const;
  virtual uint16_t readoutEventCode() const;
  virtual uint16_t delayMode() const;
  virtual uint32_t frameSize() const;
  virtual uint32_t numPixelsX() const;
  virtual uint32_t numPixelsY() const;
  virtual uint32_t numPixels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Princeton::ns_ConfigV1_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::ConfigV1> > make_ConfigV1(int version, hdf5pp::Group group, hsize_t idx);

namespace ns_ConfigV2_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint32_t width; 
  uint32_t height; 
  uint32_t orgX; 
  uint32_t orgY; 
  uint32_t binX; 
  uint32_t binY; 
  float exposureTime; 
  float coolingTemp; 
  uint16_t gainIndex; 
  uint16_t readoutSpeedIndex; 
  uint16_t readoutEventCode; 
  uint16_t delayMode; 
  uint32_t frameSize; 
  uint32_t numPixelsX; 
  uint32_t numPixelsY; 
  uint32_t numPixels; 

};
}


class ConfigV2_v0 : public Psana::Princeton::ConfigV2 {
public:
  typedef Psana::Princeton::ConfigV2 PsanaType;
  ConfigV2_v0() {}
  ConfigV2_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  ConfigV2_v0(const boost::shared_ptr<Princeton::ns_ConfigV2_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~ConfigV2_v0() {}
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t orgX() const;
  virtual uint32_t orgY() const;
  virtual uint32_t binX() const;
  virtual uint32_t binY() const;
  virtual float exposureTime() const;
  virtual float coolingTemp() const;
  virtual uint16_t gainIndex() const;
  virtual uint16_t readoutSpeedIndex() const;
  virtual uint16_t readoutEventCode() const;
  virtual uint16_t delayMode() const;
  virtual uint32_t frameSize() const;
  virtual uint32_t numPixelsX() const;
  virtual uint32_t numPixelsY() const;
  virtual uint32_t numPixels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Princeton::ns_ConfigV2_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::ConfigV2> > make_ConfigV2(int version, hdf5pp::Group group, hsize_t idx);

namespace ns_ConfigV3_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint32_t width; 
  uint32_t height; 
  uint32_t orgX; 
  uint32_t orgY; 
  uint32_t binX; 
  uint32_t binY; 
  float exposureTime; 
  float coolingTemp; 
  uint8_t gainIndex; 
  uint8_t readoutSpeedIndex; 
  uint16_t exposureEventCode; 
  uint32_t numDelayShots; 
  uint32_t frameSize; 
  uint32_t numPixelsX; 
  uint32_t numPixelsY; 
  uint32_t numPixels; 

};
}


class ConfigV3_v0 : public Psana::Princeton::ConfigV3 {
public:
  typedef Psana::Princeton::ConfigV3 PsanaType;
  ConfigV3_v0() {}
  ConfigV3_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  ConfigV3_v0(const boost::shared_ptr<Princeton::ns_ConfigV3_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~ConfigV3_v0() {}
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t orgX() const;
  virtual uint32_t orgY() const;
  virtual uint32_t binX() const;
  virtual uint32_t binY() const;
  virtual float exposureTime() const;
  virtual float coolingTemp() const;
  virtual uint8_t gainIndex() const;
  virtual uint8_t readoutSpeedIndex() const;
  virtual uint16_t exposureEventCode() const;
  virtual uint32_t numDelayShots() const;
  virtual uint32_t frameSize() const;
  virtual uint32_t numPixelsX() const;
  virtual uint32_t numPixelsY() const;
  virtual uint32_t numPixels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Princeton::ns_ConfigV3_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::ConfigV3> > make_ConfigV3(int version, hdf5pp::Group group, hsize_t idx);

namespace ns_ConfigV4_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint32_t width; 
  uint32_t height; 
  uint32_t orgX; 
  uint32_t orgY; 
  uint32_t binX; 
  uint32_t binY; 
  uint32_t maskedHeight; 
  uint32_t kineticHeight; 
  float vsSpeed; 
  float exposureTime; 
  float coolingTemp; 
  uint8_t gainIndex; 
  uint8_t readoutSpeedIndex; 
  uint16_t exposureEventCode; 
  uint32_t numDelayShots; 
  uint32_t frameSize; 
  uint32_t numPixelsX; 
  uint32_t numPixelsY; 
  uint32_t numPixels; 

};
}


class ConfigV4_v0 : public Psana::Princeton::ConfigV4 {
public:
  typedef Psana::Princeton::ConfigV4 PsanaType;
  ConfigV4_v0() {}
  ConfigV4_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  ConfigV4_v0(const boost::shared_ptr<Princeton::ns_ConfigV4_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~ConfigV4_v0() {}
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t orgX() const;
  virtual uint32_t orgY() const;
  virtual uint32_t binX() const;
  virtual uint32_t binY() const;
  virtual uint32_t maskedHeight() const;
  virtual uint32_t kineticHeight() const;
  virtual float vsSpeed() const;
  virtual float exposureTime() const;
  virtual float coolingTemp() const;
  virtual uint8_t gainIndex() const;
  virtual uint8_t readoutSpeedIndex() const;
  virtual uint16_t exposureEventCode() const;
  virtual uint32_t numDelayShots() const;
  virtual uint32_t frameSize() const;
  virtual uint32_t numPixelsX() const;
  virtual uint32_t numPixelsY() const;
  virtual uint32_t numPixels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Princeton::ns_ConfigV4_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::ConfigV4> > make_ConfigV4(int version, hdf5pp::Group group, hsize_t idx);

namespace ns_ConfigV5_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint32_t width; 
  uint32_t height; 
  uint32_t orgX; 
  uint32_t orgY; 
  uint32_t binX; 
  uint32_t binY; 
  float exposureTime; 
  float coolingTemp; 
  uint16_t gainIndex; 
  uint16_t readoutSpeedIndex; 
  uint32_t maskedHeight; 
  uint32_t kineticHeight; 
  float vsSpeed; 
  int16_t infoReportInterval; 
  uint16_t exposureEventCode; 
  uint32_t numDelayShots; 
  uint32_t frameSize; 
  uint32_t numPixelsX; 
  uint32_t numPixelsY; 
  uint32_t numPixels; 

};
}


class ConfigV5_v0 : public Psana::Princeton::ConfigV5 {
public:
  typedef Psana::Princeton::ConfigV5 PsanaType;
  ConfigV5_v0() {}
  ConfigV5_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  ConfigV5_v0(const boost::shared_ptr<Princeton::ns_ConfigV5_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~ConfigV5_v0() {}
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t orgX() const;
  virtual uint32_t orgY() const;
  virtual uint32_t binX() const;
  virtual uint32_t binY() const;
  virtual float exposureTime() const;
  virtual float coolingTemp() const;
  virtual uint16_t gainIndex() const;
  virtual uint16_t readoutSpeedIndex() const;
  virtual uint32_t maskedHeight() const;
  virtual uint32_t kineticHeight() const;
  virtual float vsSpeed() const;
  virtual int16_t infoReportInterval() const;
  virtual uint16_t exposureEventCode() const;
  virtual uint32_t numDelayShots() const;
  virtual uint32_t frameSize() const;
  virtual uint32_t numPixelsX() const;
  virtual uint32_t numPixelsY() const;
  virtual uint32_t numPixels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Princeton::ns_ConfigV5_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::ConfigV5> > make_ConfigV5(int version, hdf5pp::Group group, hsize_t idx);

namespace ns_FrameV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  ~dataset_data();

  uint32_t shotIdStart; 
  float readoutTime; 

};
}


template <typename Config>
class FrameV1_v0 : public Psana::Princeton::FrameV1 {
public:
  typedef Psana::Princeton::FrameV1 PsanaType;
  FrameV1_v0() {}
  FrameV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~FrameV1_v0() {}
  virtual uint32_t shotIdStart() const;
  virtual float readoutTime() const;
  virtual ndarray<const uint16_t, 2> data() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;
  mutable boost::shared_ptr<Princeton::ns_FrameV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable ndarray<const uint16_t, 2> m_ds_image;
  void read_ds_image() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV1>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV2>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV3>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV4>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV5>& cfg);

namespace ns_FrameV2_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  ~dataset_data();

  uint32_t shotIdStart; 
  float readoutTime; 
  float temperature; 

};
}


template <typename Config>
class FrameV2_v0 : public Psana::Princeton::FrameV2 {
public:
  typedef Psana::Princeton::FrameV2 PsanaType;
  FrameV2_v0() {}
  FrameV2_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~FrameV2_v0() {}
  virtual uint32_t shotIdStart() const;
  virtual float readoutTime() const;
  virtual float temperature() const;
  virtual ndarray<const uint16_t, 2> data() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;
  mutable boost::shared_ptr<Princeton::ns_FrameV2_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable ndarray<const uint16_t, 2> m_ds_image;
  void read_ds_image() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV2> > make_FrameV2(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV1>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV2> > make_FrameV2(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV2>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV2> > make_FrameV2(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV3>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV2> > make_FrameV2(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV4>& cfg);
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::FrameV2> > make_FrameV2(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Princeton::ConfigV5>& cfg);

namespace ns_InfoV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  ~dataset_data();

  float temperature; 

  operator Psana::Princeton::InfoV1() const { return Psana::Princeton::InfoV1(temperature); }
};
}
class Proxy_InfoV1_v0 : public PSEvt::Proxy<Psana::Princeton::InfoV1> {
public:
  typedef Psana::Princeton::InfoV1 PsanaType;

  Proxy_InfoV1_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_InfoV1_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Princeton::InfoV1> > make_InfoV1(int version, hdf5pp::Group group, hsize_t idx);
} // namespace Princeton
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_PRINCETON_DDL_H