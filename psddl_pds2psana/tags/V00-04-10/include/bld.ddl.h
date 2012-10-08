#ifndef PSDDL_PDS2PSANA_BLD_DDL_H
#define PSDDL_PDS2PSANA_BLD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <boost/shared_ptr.hpp>
#include "psddl_psana/bld.ddl.h"
#include "psddl_pdsdata/bld.ddl.h"
#include "psddl_pds2psana/camera.ddl.h"
#include "psddl_pds2psana/ipimb.ddl.h"
#include "psddl_pds2psana/lusi.ddl.h"
#include "psddl_pds2psana/pulnix.ddl.h"
namespace psddl_pds2psana {
namespace Bld {
Psana::Bld::BldDataFEEGasDetEnergy pds_to_psana(PsddlPds::Bld::BldDataFEEGasDetEnergy pds);

Psana::Bld::BldDataEBeamV0 pds_to_psana(PsddlPds::Bld::BldDataEBeamV0 pds);

Psana::Bld::BldDataEBeamV1 pds_to_psana(PsddlPds::Bld::BldDataEBeamV1 pds);

Psana::Bld::BldDataEBeamV2 pds_to_psana(PsddlPds::Bld::BldDataEBeamV2 pds);

Psana::Bld::BldDataEBeamV3 pds_to_psana(PsddlPds::Bld::BldDataEBeamV3 pds);

Psana::Bld::BldDataPhaseCavity pds_to_psana(PsddlPds::Bld::BldDataPhaseCavity pds);


class BldDataIpimbV0 : public Psana::Bld::BldDataIpimbV0 {
public:
  typedef PsddlPds::Bld::BldDataIpimbV0 XtcType;
  typedef Psana::Bld::BldDataIpimbV0 PsanaType;
  BldDataIpimbV0(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~BldDataIpimbV0();
  virtual const Psana::Ipimb::DataV1& ipimbData() const;
  virtual const Psana::Ipimb::ConfigV1& ipimbConfig() const;
  virtual const Psana::Lusi::IpmFexV1& ipmFexData() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  psddl_pds2psana::Ipimb::DataV1 _ipimbData;
  psddl_pds2psana::Ipimb::ConfigV1 _ipimbConfig;
  Psana::Lusi::IpmFexV1 _ipmFexData;
};


class BldDataIpimbV1 : public Psana::Bld::BldDataIpimbV1 {
public:
  typedef PsddlPds::Bld::BldDataIpimbV1 XtcType;
  typedef Psana::Bld::BldDataIpimbV1 PsanaType;
  BldDataIpimbV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~BldDataIpimbV1();
  virtual const Psana::Ipimb::DataV2& ipimbData() const;
  virtual const Psana::Ipimb::ConfigV2& ipimbConfig() const;
  virtual const Psana::Lusi::IpmFexV1& ipmFexData() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  psddl_pds2psana::Ipimb::DataV2 _ipimbData;
  psddl_pds2psana::Ipimb::ConfigV2 _ipimbConfig;
  Psana::Lusi::IpmFexV1 _ipmFexData;
};


class BldDataPimV1 : public Psana::Bld::BldDataPimV1 {
public:
  typedef PsddlPds::Bld::BldDataPimV1 XtcType;
  typedef Psana::Bld::BldDataPimV1 PsanaType;
  BldDataPimV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~BldDataPimV1();
  virtual const Psana::Pulnix::TM6740ConfigV2& camConfig() const;
  virtual const Psana::Lusi::PimImageConfigV1& pimConfig() const;
  virtual const Psana::Camera::FrameV1& frame() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  psddl_pds2psana::Pulnix::TM6740ConfigV2 _camConfig;
  Psana::Lusi::PimImageConfigV1 _pimConfig;
  psddl_pds2psana::Camera::FrameV1 _frame;
};


class BldDataGMDV0 : public Psana::Bld::BldDataGMDV0 {
public:
  typedef PsddlPds::Bld::BldDataGMDV0 XtcType;
  typedef Psana::Bld::BldDataGMDV0 PsanaType;
  BldDataGMDV0(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~BldDataGMDV0();
  virtual const char* gasType() const;
  virtual double pressure() const;
  virtual double temperature() const;
  virtual double current() const;
  virtual int32_t hvMeshElectron() const;
  virtual int32_t hvMeshIon() const;
  virtual int32_t hvMultIon() const;
  virtual double chargeQ() const;
  virtual double photonEnergy() const;
  virtual double photonsPerPulse() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
};

} // namespace Bld
} // namespace psddl_pds2psana
#endif // PSDDL_PDS2PSANA_BLD_DDL_H
