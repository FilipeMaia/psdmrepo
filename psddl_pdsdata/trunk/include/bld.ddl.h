#ifndef PSDDLPDS_BLD_DDL_H
#define PSDDLPDS_BLD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

#include <cstddef>

#include "psddl_pdsdata/ipimb.ddl.h"
#include "psddl_pdsdata/lusi.ddl.h"
namespace PsddlPds {
namespace Bld {

/** @class BldDataFEEGasDetEnergy

  Four energy measurements from Front End Enclosure Gas Detector.
	       	PV names: GDET:FEE1:11:ENRC, GDET:FEE1:12:ENRC, GDET:FEE1:21:ENRC, GDET:FEE1:22:ENRC.
*/

#pragma pack(push,4)

class BldDataFEEGasDetEnergy {
public:
  enum {
    Version = 0 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_FEEGasDetEnergy /**< XTC type ID value (from Pds::TypeId class) */
  };
  BldDataFEEGasDetEnergy(double arg__f_11_ENRC, double arg__f_12_ENRC, double arg__f_21_ENRC, double arg__f_22_ENRC)
    : _f_11_ENRC(arg__f_11_ENRC), _f_12_ENRC(arg__f_12_ENRC), _f_21_ENRC(arg__f_21_ENRC), _f_22_ENRC(arg__f_22_ENRC)
  {
  }
  /** Value of GDET:FEE1:11:ENRC, in mJ. */
  double f_11_ENRC() const {return _f_11_ENRC;}
  /** Value of GDET:FEE1:12:ENRC, in mJ. */
  double f_12_ENRC() const {return _f_12_ENRC;}
  /** Value of GDET:FEE1:21:ENRC, in mJ. */
  double f_21_ENRC() const {return _f_21_ENRC;}
  /** Value of GDET:FEE1:22:ENRC, in mJ. */
  double f_22_ENRC() const {return _f_22_ENRC;}
  static uint32_t _sizeof()  {return 32;}
private:
  double	_f_11_ENRC;	/**< Value of GDET:FEE1:11:ENRC, in mJ. */
  double	_f_12_ENRC;	/**< Value of GDET:FEE1:12:ENRC, in mJ. */
  double	_f_21_ENRC;	/**< Value of GDET:FEE1:21:ENRC, in mJ. */
  double	_f_22_ENRC;	/**< Value of GDET:FEE1:22:ENRC, in mJ. */
};
#pragma pack(pop)

/** @class BldDataEBeamV0

  Beam parameters.
*/

#pragma pack(push,4)

class BldDataEBeamV0 {
public:
  enum {
    Version = 0 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EBeam /**< XTC type ID value (from Pds::TypeId class) */
  };
  BldDataEBeamV0(uint32_t arg__uDamageMask, double arg__fEbeamCharge, double arg__fEbeamL3Energy, double arg__fEbeamLTUPosX, double arg__fEbeamLTUPosY, double arg__fEbeamLTUAngX, double arg__fEbeamLTUAngY)
    : _uDamageMask(arg__uDamageMask), _fEbeamCharge(arg__fEbeamCharge), _fEbeamL3Energy(arg__fEbeamL3Energy), _fEbeamLTUPosX(arg__fEbeamLTUPosX), _fEbeamLTUPosY(arg__fEbeamLTUPosY), _fEbeamLTUAngX(arg__fEbeamLTUAngX), _fEbeamLTUAngY(arg__fEbeamLTUAngY)
  {
  }
  /** Damage mask. */
  uint32_t damageMask() const {return _uDamageMask;}
  /** Beam charge in nC. */
  double ebeamCharge() const {return _fEbeamCharge;}
  /** Beam energy in MeV. */
  double ebeamL3Energy() const {return _fEbeamL3Energy;}
  /** LTU beam position in mm. */
  double ebeamLTUPosX() const {return _fEbeamLTUPosX;}
  /** LTU beam position in mm. */
  double ebeamLTUPosY() const {return _fEbeamLTUPosY;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngX() const {return _fEbeamLTUAngX;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngY() const {return _fEbeamLTUAngY;}
  static uint32_t _sizeof()  {return 52;}
private:
  uint32_t	_uDamageMask;	/**< Damage mask. */
  double	_fEbeamCharge;	/**< Beam charge in nC. */
  double	_fEbeamL3Energy;	/**< Beam energy in MeV. */
  double	_fEbeamLTUPosX;	/**< LTU beam position in mm. */
  double	_fEbeamLTUPosY;	/**< LTU beam position in mm. */
  double	_fEbeamLTUAngX;	/**< LTU beam angle in mrad. */
  double	_fEbeamLTUAngY;	/**< LTU beam angle in mrad. */
};
#pragma pack(pop)

/** @class BldDataEBeamV1

  
*/

#pragma pack(push,4)

class BldDataEBeamV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EBeam /**< XTC type ID value (from Pds::TypeId class) */
  };
  BldDataEBeamV1(uint32_t arg__uDamageMask, double arg__fEbeamCharge, double arg__fEbeamL3Energy, double arg__fEbeamLTUPosX, double arg__fEbeamLTUPosY, double arg__fEbeamLTUAngX, double arg__fEbeamLTUAngY, double arg__fEbeamPkCurrBC2)
    : _uDamageMask(arg__uDamageMask), _fEbeamCharge(arg__fEbeamCharge), _fEbeamL3Energy(arg__fEbeamL3Energy), _fEbeamLTUPosX(arg__fEbeamLTUPosX), _fEbeamLTUPosY(arg__fEbeamLTUPosY), _fEbeamLTUAngX(arg__fEbeamLTUAngX), _fEbeamLTUAngY(arg__fEbeamLTUAngY), _fEbeamPkCurrBC2(arg__fEbeamPkCurrBC2)
  {
  }
  /** Damage mask. */
  uint32_t damageMask() const {return _uDamageMask;}
  /** Beam charge in nC. */
  double ebeamCharge() const {return _fEbeamCharge;}
  /** Beam energy in MeV. */
  double ebeamL3Energy() const {return _fEbeamL3Energy;}
  /** LTU beam position in mm. */
  double ebeamLTUPosX() const {return _fEbeamLTUPosX;}
  /** LTU beam position in mm. */
  double ebeamLTUPosY() const {return _fEbeamLTUPosY;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngX() const {return _fEbeamLTUAngX;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngY() const {return _fEbeamLTUAngY;}
  /** Beam current in Amps. */
  double ebeamPkCurrBC2() const {return _fEbeamPkCurrBC2;}
  static uint32_t _sizeof()  {return 60;}
private:
  uint32_t	_uDamageMask;	/**< Damage mask. */
  double	_fEbeamCharge;	/**< Beam charge in nC. */
  double	_fEbeamL3Energy;	/**< Beam energy in MeV. */
  double	_fEbeamLTUPosX;	/**< LTU beam position in mm. */
  double	_fEbeamLTUPosY;	/**< LTU beam position in mm. */
  double	_fEbeamLTUAngX;	/**< LTU beam angle in mrad. */
  double	_fEbeamLTUAngY;	/**< LTU beam angle in mrad. */
  double	_fEbeamPkCurrBC2;	/**< Beam current in Amps. */
};
#pragma pack(pop)

/** @class BldDataEBeamV2

  
*/

#pragma pack(push,4)

class BldDataEBeamV2 {
public:
  enum {
    Version = 2 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_EBeam /**< XTC type ID value (from Pds::TypeId class) */
  };
  BldDataEBeamV2(uint32_t arg__uDamageMask, double arg__fEbeamCharge, double arg__fEbeamL3Energy, double arg__fEbeamLTUPosX, double arg__fEbeamLTUPosY, double arg__fEbeamLTUAngX, double arg__fEbeamLTUAngY, double arg__fEbeamPkCurrBC2, double arg__fEbeamEnergyBC2)
    : _uDamageMask(arg__uDamageMask), _fEbeamCharge(arg__fEbeamCharge), _fEbeamL3Energy(arg__fEbeamL3Energy), _fEbeamLTUPosX(arg__fEbeamLTUPosX), _fEbeamLTUPosY(arg__fEbeamLTUPosY), _fEbeamLTUAngX(arg__fEbeamLTUAngX), _fEbeamLTUAngY(arg__fEbeamLTUAngY), _fEbeamPkCurrBC2(arg__fEbeamPkCurrBC2), _fEbeamEnergyBC2(arg__fEbeamEnergyBC2)
  {
  }
  /** Damage mask. */
  uint32_t damageMask() const {return _uDamageMask;}
  /** Beam charge in nC. */
  double ebeamCharge() const {return _fEbeamCharge;}
  /** Beam energy in MeV. */
  double ebeamL3Energy() const {return _fEbeamL3Energy;}
  /** LTU beam position in mm. */
  double ebeamLTUPosX() const {return _fEbeamLTUPosX;}
  /** LTU beam position in mm. */
  double ebeamLTUPosY() const {return _fEbeamLTUPosY;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngX() const {return _fEbeamLTUAngX;}
  /** LTU beam angle in mrad. */
  double ebeamLTUAngY() const {return _fEbeamLTUAngY;}
  /** Beam current in Amps. */
  double ebeamPkCurrBC2() const {return _fEbeamPkCurrBC2;}
  /** Beam energy in MeV. */
  double ebeamEnergyBC2() const {return _fEbeamEnergyBC2;}
  static uint32_t _sizeof()  {return 68;}
private:
  uint32_t	_uDamageMask;	/**< Damage mask. */
  double	_fEbeamCharge;	/**< Beam charge in nC. */
  double	_fEbeamL3Energy;	/**< Beam energy in MeV. */
  double	_fEbeamLTUPosX;	/**< LTU beam position in mm. */
  double	_fEbeamLTUPosY;	/**< LTU beam position in mm. */
  double	_fEbeamLTUAngX;	/**< LTU beam angle in mrad. */
  double	_fEbeamLTUAngY;	/**< LTU beam angle in mrad. */
  double	_fEbeamPkCurrBC2;	/**< Beam current in Amps. */
  double	_fEbeamEnergyBC2;	/**< Beam energy in MeV. */
};
#pragma pack(pop)

/** @class BldDataPhaseCavity

  PV names: UND:R02:IOC:16:BAT:FitTime1, UND:R02:IOC:16:BAT:FitTime2, 
                UND:R02:IOC:16:BAT:Charge1,  UND:R02:IOC:16:BAT:Charge2
*/

#pragma pack(push,4)

class BldDataPhaseCavity {
public:
  enum {
    Version = 0 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_PhaseCavity /**< XTC type ID value (from Pds::TypeId class) */
  };
  BldDataPhaseCavity(double arg__fFitTime1, double arg__fFitTime2, double arg__fCharge1, double arg__fCharge2)
    : _fFitTime1(arg__fFitTime1), _fFitTime2(arg__fFitTime2), _fCharge1(arg__fCharge1), _fCharge2(arg__fCharge2)
  {
  }
  /** UND:R02:IOC:16:BAT:FitTime1 value in pico-seconds. */
  double fitTime1() const {return _fFitTime1;}
  /** UND:R02:IOC:16:BAT:FitTime2 value in pico-seconds. */
  double fitTime2() const {return _fFitTime2;}
  /** UND:R02:IOC:16:BAT:Charge1 value in pico-columbs. */
  double charge1() const {return _fCharge1;}
  /** UND:R02:IOC:16:BAT:Charge2 value in pico-columbs. */
  double charge2() const {return _fCharge2;}
  static uint32_t _sizeof()  {return 32;}
private:
  double	_fFitTime1;	/**< UND:R02:IOC:16:BAT:FitTime1 value in pico-seconds. */
  double	_fFitTime2;	/**< UND:R02:IOC:16:BAT:FitTime2 value in pico-seconds. */
  double	_fCharge1;	/**< UND:R02:IOC:16:BAT:Charge1 value in pico-columbs. */
  double	_fCharge2;	/**< UND:R02:IOC:16:BAT:Charge2 value in pico-columbs. */
};
#pragma pack(pop)

/** @class BldDataIpimbV0

  Combined structure which includes Ipimb.DataV1, Ipimb.ConfigV1, and 
            Lusi.IpmFexV1 objects.
*/

#pragma pack(push,4)

class BldDataIpimbV0 {
public:
  enum {
    Version = 0 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_SharedIpimb /**< XTC type ID value (from Pds::TypeId class) */
  };
  const Ipimb::DataV1& ipimbData() const {return _ipimbData;}
  const Ipimb::ConfigV1& ipimbConfig() const {return _ipimbConfig;}
  const Lusi::IpmFexV1& ipmFexData() const {return _ipmFexData;}
  static uint32_t _sizeof()  {return ((0+(Ipimb::DataV1::_sizeof()))+(Ipimb::ConfigV1::_sizeof()))+(Lusi::IpmFexV1::_sizeof());}
private:
  Ipimb::DataV1	_ipimbData;
  Ipimb::ConfigV1	_ipimbConfig;
  Lusi::IpmFexV1	_ipmFexData;
};
#pragma pack(pop)

/** @class BldDataIpimbV1

  Combined structure which includes Ipimb.DataV2, Ipimb.ConfigV2, and 
            Lusi.IpmFexV1 objects.
*/

#pragma pack(push,4)

class BldDataIpimbV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_SharedIpimb /**< XTC type ID value (from Pds::TypeId class) */
  };
  const Ipimb::DataV2& ipimbData() const {return _ipimbData;}
  const Ipimb::ConfigV2& ipimbConfig() const {return _ipimbConfig;}
  const Lusi::IpmFexV1& ipmFexData() const {return _ipmFexData;}
  static uint32_t _sizeof()  {return ((0+(Ipimb::DataV2::_sizeof()))+(Ipimb::ConfigV2::_sizeof()))+(Lusi::IpmFexV1::_sizeof());}
private:
  Ipimb::DataV2	_ipimbData;
  Ipimb::ConfigV2	_ipimbConfig;
  Lusi::IpmFexV1	_ipmFexData;
};
#pragma pack(pop)
} // namespace Bld
} // namespace PsddlPds
#endif // PSDDLPDS_BLD_DDL_H
