#ifndef PSDDL_PDS2PSANA_EPICS_DDL_H
#define PSDDL_PDS2PSANA_EPICS_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <boost/shared_ptr.hpp>
#include "psddl_psana/epics.ddl.h"
#include "pdsdata/psddl/epics.ddl.h"
namespace psddl_pds2psana {
namespace Epics {
Psana::Epics::epicsTimeStamp pds_to_psana(Pds::Epics::epicsTimeStamp pds);

Psana::Epics::dbr_time_string pds_to_psana(Pds::Epics::dbr_time_string pds);

Psana::Epics::dbr_time_short pds_to_psana(Pds::Epics::dbr_time_short pds);

Psana::Epics::dbr_time_float pds_to_psana(Pds::Epics::dbr_time_float pds);

Psana::Epics::dbr_time_enum pds_to_psana(Pds::Epics::dbr_time_enum pds);

Psana::Epics::dbr_time_char pds_to_psana(Pds::Epics::dbr_time_char pds);

Psana::Epics::dbr_time_long pds_to_psana(Pds::Epics::dbr_time_long pds);

Psana::Epics::dbr_time_double pds_to_psana(Pds::Epics::dbr_time_double pds);

Psana::Epics::dbr_sts_string pds_to_psana(Pds::Epics::dbr_sts_string pds);

Psana::Epics::dbr_ctrl_short pds_to_psana(Pds::Epics::dbr_ctrl_short pds);

Psana::Epics::dbr_ctrl_float pds_to_psana(Pds::Epics::dbr_ctrl_float pds);

Psana::Epics::dbr_ctrl_enum pds_to_psana(Pds::Epics::dbr_ctrl_enum pds);

Psana::Epics::dbr_ctrl_char pds_to_psana(Pds::Epics::dbr_ctrl_char pds);

Psana::Epics::dbr_ctrl_long pds_to_psana(Pds::Epics::dbr_ctrl_long pds);

Psana::Epics::dbr_ctrl_double pds_to_psana(Pds::Epics::dbr_ctrl_double pds);


class EpicsPvHeader : public Psana::Epics::EpicsPvHeader {
public:
  typedef Pds::Epics::EpicsPvHeader XtcType;
  typedef Psana::Epics::EpicsPvHeader PsanaType;
  EpicsPvHeader(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvHeader();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class EpicsPvCtrlHeader : public Psana::Epics::EpicsPvCtrlHeader {
public:
  typedef Pds::Epics::EpicsPvCtrlHeader XtcType;
  typedef Psana::Epics::EpicsPvCtrlHeader PsanaType;
  EpicsPvCtrlHeader(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlHeader();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class EpicsPvTimeHeader : public Psana::Epics::EpicsPvTimeHeader {
public:
  typedef Pds::Epics::EpicsPvTimeHeader XtcType;
  typedef Psana::Epics::EpicsPvTimeHeader PsanaType;
  EpicsPvTimeHeader(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeHeader();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};


class EpicsPvCtrlString : public Psana::Epics::EpicsPvCtrlString {
public:
  typedef Pds::Epics::EpicsPvCtrlString XtcType;
  typedef Psana::Epics::EpicsPvCtrlString PsanaType;
  EpicsPvCtrlString(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlString();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_sts_string& dbr() const;
  virtual const char* data(uint32_t i0) const;
  virtual const char* value(uint32_t i) const;
  virtual std::vector<int> data_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_sts_string _dbr;
};


class EpicsPvCtrlShort : public Psana::Epics::EpicsPvCtrlShort {
public:
  typedef Pds::Epics::EpicsPvCtrlShort XtcType;
  typedef Psana::Epics::EpicsPvCtrlShort PsanaType;
  EpicsPvCtrlShort(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlShort();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_short& dbr() const;
  virtual ndarray<const int16_t, 1> data() const;
  virtual int16_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_short _dbr;
};


class EpicsPvCtrlFloat : public Psana::Epics::EpicsPvCtrlFloat {
public:
  typedef Pds::Epics::EpicsPvCtrlFloat XtcType;
  typedef Psana::Epics::EpicsPvCtrlFloat PsanaType;
  EpicsPvCtrlFloat(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlFloat();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_float& dbr() const;
  virtual ndarray<const float, 1> data() const;
  virtual float value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_float _dbr;
};


class EpicsPvCtrlEnum : public Psana::Epics::EpicsPvCtrlEnum {
public:
  typedef Pds::Epics::EpicsPvCtrlEnum XtcType;
  typedef Psana::Epics::EpicsPvCtrlEnum PsanaType;
  EpicsPvCtrlEnum(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlEnum();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_enum& dbr() const;
  virtual ndarray<const uint16_t, 1> data() const;
  virtual uint16_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_enum _dbr;
};


class EpicsPvCtrlChar : public Psana::Epics::EpicsPvCtrlChar {
public:
  typedef Pds::Epics::EpicsPvCtrlChar XtcType;
  typedef Psana::Epics::EpicsPvCtrlChar PsanaType;
  EpicsPvCtrlChar(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlChar();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_char& dbr() const;
  virtual ndarray<const uint8_t, 1> data() const;
  virtual uint8_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_char _dbr;
};


class EpicsPvCtrlLong : public Psana::Epics::EpicsPvCtrlLong {
public:
  typedef Pds::Epics::EpicsPvCtrlLong XtcType;
  typedef Psana::Epics::EpicsPvCtrlLong PsanaType;
  EpicsPvCtrlLong(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlLong();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_long& dbr() const;
  virtual ndarray<const int32_t, 1> data() const;
  virtual int32_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_long _dbr;
};


class EpicsPvCtrlDouble : public Psana::Epics::EpicsPvCtrlDouble {
public:
  typedef Pds::Epics::EpicsPvCtrlDouble XtcType;
  typedef Psana::Epics::EpicsPvCtrlDouble PsanaType;
  EpicsPvCtrlDouble(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvCtrlDouble();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual const char* pvName() const;
  virtual const Psana::Epics::dbr_ctrl_double& dbr() const;
  virtual ndarray<const double, 1> data() const;
  virtual double value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_ctrl_double _dbr;
};


class EpicsPvTimeString : public Psana::Epics::EpicsPvTimeString {
public:
  typedef Pds::Epics::EpicsPvTimeString XtcType;
  typedef Psana::Epics::EpicsPvTimeString PsanaType;
  EpicsPvTimeString(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeString();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_string& dbr() const;
  virtual const char* data(uint32_t i0) const;
  virtual const char* value(uint32_t i) const;
  virtual std::vector<int> data_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_string _dbr;
};


class EpicsPvTimeShort : public Psana::Epics::EpicsPvTimeShort {
public:
  typedef Pds::Epics::EpicsPvTimeShort XtcType;
  typedef Psana::Epics::EpicsPvTimeShort PsanaType;
  EpicsPvTimeShort(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeShort();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_short& dbr() const;
  virtual ndarray<const int16_t, 1> data() const;
  virtual int16_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_short _dbr;
};


class EpicsPvTimeFloat : public Psana::Epics::EpicsPvTimeFloat {
public:
  typedef Pds::Epics::EpicsPvTimeFloat XtcType;
  typedef Psana::Epics::EpicsPvTimeFloat PsanaType;
  EpicsPvTimeFloat(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeFloat();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_float& dbr() const;
  virtual ndarray<const float, 1> data() const;
  virtual float value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_float _dbr;
};


class EpicsPvTimeEnum : public Psana::Epics::EpicsPvTimeEnum {
public:
  typedef Pds::Epics::EpicsPvTimeEnum XtcType;
  typedef Psana::Epics::EpicsPvTimeEnum PsanaType;
  EpicsPvTimeEnum(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeEnum();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_enum& dbr() const;
  virtual ndarray<const uint16_t, 1> data() const;
  virtual uint16_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_enum _dbr;
};


class EpicsPvTimeChar : public Psana::Epics::EpicsPvTimeChar {
public:
  typedef Pds::Epics::EpicsPvTimeChar XtcType;
  typedef Psana::Epics::EpicsPvTimeChar PsanaType;
  EpicsPvTimeChar(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeChar();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_char& dbr() const;
  virtual ndarray<const uint8_t, 1> data() const;
  virtual uint8_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_char _dbr;
};


class EpicsPvTimeLong : public Psana::Epics::EpicsPvTimeLong {
public:
  typedef Pds::Epics::EpicsPvTimeLong XtcType;
  typedef Psana::Epics::EpicsPvTimeLong PsanaType;
  EpicsPvTimeLong(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeLong();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_long& dbr() const;
  virtual ndarray<const int32_t, 1> data() const;
  virtual int32_t value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_long _dbr;
};


class EpicsPvTimeDouble : public Psana::Epics::EpicsPvTimeDouble {
public:
  typedef Pds::Epics::EpicsPvTimeDouble XtcType;
  typedef Psana::Epics::EpicsPvTimeDouble PsanaType;
  EpicsPvTimeDouble(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~EpicsPvTimeDouble();
  virtual int16_t pvId() const;
  virtual int16_t dbrType() const;
  virtual int16_t numElements() const;
  virtual uint8_t isCtrl() const;
  virtual uint8_t isTime() const;
  virtual uint16_t status() const;
  virtual uint16_t severity() const;
  virtual Psana::Epics::epicsTimeStamp stamp() const;
  virtual const Psana::Epics::dbr_time_double& dbr() const;
  virtual ndarray<const double, 1> data() const;
  virtual double value(uint32_t i) const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  Psana::Epics::dbr_time_double _dbr;
};

Psana::Epics::PvConfigV1 pds_to_psana(Pds::Epics::PvConfigV1 pds);


class ConfigV1 : public Psana::Epics::ConfigV1 {
public:
  typedef Pds::Epics::ConfigV1 XtcType;
  typedef Psana::Epics::ConfigV1 PsanaType;
  ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV1();
  virtual int32_t numPv() const;
  virtual ndarray<const Psana::Epics::PvConfigV1, 1> getPvConfig() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  ndarray<Psana::Epics::PvConfigV1, 1> _pvConfig_ndarray_storage_;
};

} // namespace Epics
} // namespace psddl_pds2psana
#endif // PSDDL_PDS2PSANA_EPICS_DDL_H
