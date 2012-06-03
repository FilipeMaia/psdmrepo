#ifndef PSDDL_PDS2PSANA_TIMEPIX_DDL_H
#define PSDDL_PDS2PSANA_TIMEPIX_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <boost/shared_ptr.hpp>
#include "psddl_psana/timepix.ddl.h"
#include "psddl_pdsdata/timepix.ddl.h"
namespace psddl_pds2psana {
namespace Timepix {

class ConfigV1 : public Psana::Timepix::ConfigV1 {
public:
  typedef PsddlPds::Timepix::ConfigV1 XtcType;
  typedef Psana::Timepix::ConfigV1 PsanaType;
  ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV1();
  virtual Psana::Timepix::ConfigV1::ReadoutSpeed readoutSpeed() const;
  virtual Psana::Timepix::ConfigV1::TriggerMode triggerMode() const;
  virtual int32_t shutterTimeout() const;
  virtual int32_t dac0Ikrum() const;
  virtual int32_t dac0Disc() const;
  virtual int32_t dac0Preamp() const;
  virtual int32_t dac0BufAnalogA() const;
  virtual int32_t dac0BufAnalogB() const;
  virtual int32_t dac0Hist() const;
  virtual int32_t dac0ThlFine() const;
  virtual int32_t dac0ThlCourse() const;
  virtual int32_t dac0Vcas() const;
  virtual int32_t dac0Fbk() const;
  virtual int32_t dac0Gnd() const;
  virtual int32_t dac0Ths() const;
  virtual int32_t dac0BiasLvds() const;
  virtual int32_t dac0RefLvds() const;
  virtual int32_t dac1Ikrum() const;
  virtual int32_t dac1Disc() const;
  virtual int32_t dac1Preamp() const;
  virtual int32_t dac1BufAnalogA() const;
  virtual int32_t dac1BufAnalogB() const;
  virtual int32_t dac1Hist() const;
  virtual int32_t dac1ThlFine() const;
  virtual int32_t dac1ThlCourse() const;
  virtual int32_t dac1Vcas() const;
  virtual int32_t dac1Fbk() const;
  virtual int32_t dac1Gnd() const;
  virtual int32_t dac1Ths() const;
  virtual int32_t dac1BiasLvds() const;
  virtual int32_t dac1RefLvds() const;
  virtual int32_t dac2Ikrum() const;
  virtual int32_t dac2Disc() const;
  virtual int32_t dac2Preamp() const;
  virtual int32_t dac2BufAnalogA() const;
  virtual int32_t dac2BufAnalogB() const;
  virtual int32_t dac2Hist() const;
  virtual int32_t dac2ThlFine() const;
  virtual int32_t dac2ThlCourse() const;
  virtual int32_t dac2Vcas() const;
  virtual int32_t dac2Fbk() const;
  virtual int32_t dac2Gnd() const;
  virtual int32_t dac2Ths() const;
  virtual int32_t dac2BiasLvds() const;
  virtual int32_t dac2RefLvds() const;
  virtual int32_t dac3Ikrum() const;
  virtual int32_t dac3Disc() const;
  virtual int32_t dac3Preamp() const;
  virtual int32_t dac3BufAnalogA() const;
  virtual int32_t dac3BufAnalogB() const;
  virtual int32_t dac3Hist() const;
  virtual int32_t dac3ThlFine() const;
  virtual int32_t dac3ThlCourse() const;
  virtual int32_t dac3Vcas() const;
  virtual int32_t dac3Fbk() const;
  virtual int32_t dac3Gnd() const;
  virtual int32_t dac3Ths() const;
  virtual int32_t dac3BiasLvds() const;
  virtual int32_t dac3RefLvds() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class ConfigV2 : public Psana::Timepix::ConfigV2 {
public:
  typedef PsddlPds::Timepix::ConfigV2 XtcType;
  typedef Psana::Timepix::ConfigV2 PsanaType;
  ConfigV2(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV2();
  virtual Psana::Timepix::ConfigV2::ReadoutSpeed readoutSpeed() const;
  virtual Psana::Timepix::ConfigV2::TriggerMode triggerMode() const;
  virtual int32_t timepixSpeed() const;
  virtual int32_t dac0Ikrum() const;
  virtual int32_t dac0Disc() const;
  virtual int32_t dac0Preamp() const;
  virtual int32_t dac0BufAnalogA() const;
  virtual int32_t dac0BufAnalogB() const;
  virtual int32_t dac0Hist() const;
  virtual int32_t dac0ThlFine() const;
  virtual int32_t dac0ThlCourse() const;
  virtual int32_t dac0Vcas() const;
  virtual int32_t dac0Fbk() const;
  virtual int32_t dac0Gnd() const;
  virtual int32_t dac0Ths() const;
  virtual int32_t dac0BiasLvds() const;
  virtual int32_t dac0RefLvds() const;
  virtual int32_t dac1Ikrum() const;
  virtual int32_t dac1Disc() const;
  virtual int32_t dac1Preamp() const;
  virtual int32_t dac1BufAnalogA() const;
  virtual int32_t dac1BufAnalogB() const;
  virtual int32_t dac1Hist() const;
  virtual int32_t dac1ThlFine() const;
  virtual int32_t dac1ThlCourse() const;
  virtual int32_t dac1Vcas() const;
  virtual int32_t dac1Fbk() const;
  virtual int32_t dac1Gnd() const;
  virtual int32_t dac1Ths() const;
  virtual int32_t dac1BiasLvds() const;
  virtual int32_t dac1RefLvds() const;
  virtual int32_t dac2Ikrum() const;
  virtual int32_t dac2Disc() const;
  virtual int32_t dac2Preamp() const;
  virtual int32_t dac2BufAnalogA() const;
  virtual int32_t dac2BufAnalogB() const;
  virtual int32_t dac2Hist() const;
  virtual int32_t dac2ThlFine() const;
  virtual int32_t dac2ThlCourse() const;
  virtual int32_t dac2Vcas() const;
  virtual int32_t dac2Fbk() const;
  virtual int32_t dac2Gnd() const;
  virtual int32_t dac2Ths() const;
  virtual int32_t dac2BiasLvds() const;
  virtual int32_t dac2RefLvds() const;
  virtual int32_t dac3Ikrum() const;
  virtual int32_t dac3Disc() const;
  virtual int32_t dac3Preamp() const;
  virtual int32_t dac3BufAnalogA() const;
  virtual int32_t dac3BufAnalogB() const;
  virtual int32_t dac3Hist() const;
  virtual int32_t dac3ThlFine() const;
  virtual int32_t dac3ThlCourse() const;
  virtual int32_t dac3Vcas() const;
  virtual int32_t dac3Fbk() const;
  virtual int32_t dac3Gnd() const;
  virtual int32_t dac3Ths() const;
  virtual int32_t dac3BiasLvds() const;
  virtual int32_t dac3RefLvds() const;
  virtual int32_t driverVersion() const;
  virtual uint32_t firmwareVersion() const;
  virtual uint32_t pixelThreshSize() const;
  virtual ndarray<uint8_t, 1> pixelThresh() const;
  virtual const char* chip0Name() const;
  virtual const char* chip1Name() const;
  virtual const char* chip2Name() const;
  virtual const char* chip3Name() const;
  virtual int32_t chip0ID() const;
  virtual int32_t chip1ID() const;
  virtual int32_t chip2ID() const;
  virtual int32_t chip3ID() const;
  virtual int32_t chipCount() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class DataV1 : public Psana::Timepix::DataV1 {
public:
  typedef PsddlPds::Timepix::DataV1 XtcType;
  typedef Psana::Timepix::DataV1 PsanaType;
  DataV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~DataV1();
  virtual uint32_t timestamp() const;
  virtual uint16_t frameCounter() const;
  virtual uint16_t lostRows() const;
  virtual ndarray<uint16_t, 2> data() const;
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t depth() const;
  virtual uint32_t depth_bytes() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class DataV2 : public Psana::Timepix::DataV2 {
public:
  typedef PsddlPds::Timepix::DataV2 XtcType;
  typedef Psana::Timepix::DataV2 PsanaType;
  DataV2(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~DataV2();
  virtual uint16_t width() const;
  virtual uint16_t height() const;
  virtual uint32_t timestamp() const;
  virtual uint16_t frameCounter() const;
  virtual uint16_t lostRows() const;
  virtual ndarray<uint16_t, 2> data() const;
  virtual uint32_t depth() const;
  virtual uint32_t depth_bytes() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};

} // namespace Timepix
} // namespace psddl_pds2psana
#endif // PSDDL_PDS2PSANA_TIMEPIX_DDL_H
