@include "psddldata/acqiris.ddl";
@include "psddldata/camera.ddl";
@include "psddldata/ipimb.ddl";
@include "psddldata/lusi.ddl";
@include "psddldata/pulnix.ddl";
@package Bld  {


//------------------ BldDataFEEGasDetEnergy ------------------
/* Four energy measurements from Front End Enclosure Gas Detector.
               PV names: GDET:FEE1:11:ENRC, GDET:FEE1:12:ENRC, GDET:FEE1:21:ENRC, GDET:FEE1:22:ENRC. */
@type BldDataFEEGasDetEnergy
  [[type_id(Id_FEEGasDetEnergy, 0)]]
  [[value_type]]
  [[pack(4)]]
{
  double _f_11_ENRC -> f_11_ENRC; /* Value of GDET:FEE1:11:ENRC, in mJ. */
  double _f_12_ENRC -> f_12_ENRC; /* Value of GDET:FEE1:12:ENRC, in mJ. */
  double _f_21_ENRC -> f_21_ENRC; /* Value of GDET:FEE1:21:ENRC, in mJ. */
  double _f_22_ENRC -> f_22_ENRC; /* Value of GDET:FEE1:22:ENRC, in mJ. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];



}

//------------------ BldDataFEEGasDetEnergyV1 ------------------
/* Six energy measurements from Front End Enclosure Gas Detector.
   PV names: GDET:FEE1:241:ENRC, GDET:FEE1:242:ENRC, 
	GDET:FEE1:361:ENRC, GDET:FEE1:362:ENRC, 
	GDET:FEE1:363:ENRC, and GDET:FEE1:364:ENRC 
   Each pair of methods (e.g. f_11_ENRC(), f_12_ENRC() contains
   identical measurements using two different phototubes.  "11" and "12"
   are before the gas attenuation.  "21" and "22" are after gas
   attenuation.
   "63" and "64" are duplicate measurements of "21" and "22" respectively. 
    The difference is that they cover a smaller (10%) dynamic range. 
    When the beam is weak, 361 and 362 don't have good S/N, these 2 extra PVs should be used instead.  Dehong Zhang suggests that the threshold
    for "weak" is around 0.5 mJ.  */

@type BldDataFEEGasDetEnergyV1
  [[type_id(Id_FEEGasDetEnergy, 1)]]
  [[value_type]]
  [[pack(4)]]
{
  double _f_11_ENRC -> f_11_ENRC; /* First energy measurement (mJ) before attenuation. (pv name GDET:FEE1:241:ENRC) */
  double _f_12_ENRC -> f_12_ENRC; /* Second (duplicate!) energy measurement (mJ) after attenuation. (pv name GDET:FEE1:242:ENRC) */
  double _f_21_ENRC -> f_21_ENRC; /* First energy measurement (mJ) after attenuation. (pv name  GDET:FEE1:361:ENRC) */
  double _f_22_ENRC -> f_22_ENRC; /* Second (duplicate!) energy measurement (mJ) after attenuation. (pv name GDET:FEE1:362:ENRC)*/
  double _f_63_ENRC -> f_63_ENRC; /* First energy measurement (mJ) for small signals (<0.5 mJ), after attenuation. (pv name GDET:FEE1:363:ENRC) */
  double _f_64_ENRC -> f_64_ENRC; /* Second (duplicate!) energy measurement (mJ) for small signals (<0.5mJ), after attenutation. (pv name GDET:FEE1:364:ENRC) */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV0 ------------------
/* Beam parameters. */
@type BldDataEBeamV0
  [[type_id(Id_EBeam, 0)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV1 ------------------
@type BldDataEBeamV1
  [[type_id(Id_EBeam, 1)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV2 ------------------
@type BldDataEBeamV2
  [[type_id(Id_EBeam, 2)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam energy in MeV. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV3 ------------------
@type BldDataEBeamV3
  [[type_id(Id_EBeam, 3)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
    EbeamPkCurrBC1Damage = 0x100,
    EbeamEnergyBC1Damage = 0x200,
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;  /* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;  /* Beam position in mm (related to beam energy). */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV4 ------------------
@type BldDataEBeamV4
  [[type_id(Id_EBeam, 4)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
    EbeamPkCurrBC1Damage = 0x100,
    EbeamEnergyBC1Damage = 0x200,
    EbeamUndPosXDamage = 0x400,
    EbeamUndPosYDamage = 0x800,
    EbeamUndAngXDamage = 0x1000,
    EbeamUndAngYDamage = 0x2000,
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;  /* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;  /* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;  /* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;  /* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;  /* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;  /* Undulator launch feedback beam y-angle in mrad. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV5 ------------------
@type BldDataEBeamV5
  [[type_id(Id_EBeam, 5)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
    EbeamPkCurrBC1Damage = 0x100,
    EbeamEnergyBC1Damage = 0x200,
    EbeamUndPosXDamage = 0x400,
    EbeamUndPosYDamage = 0x800,
    EbeamUndAngXDamage = 0x1000,
    EbeamUndAngYDamage = 0x2000,
    EbeamXTCAVAmplDamage  = 0x4000,
    EbeamXTCAVPhaseDamage = 0x8000,
    EbeamDumpChargeDamage = 0x10000
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;  /* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;  /* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;  /* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;  /* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;  /* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;  /* Undulator launch feedback beam y-angle in mrad. */
  double _fEbeamXTCAVAmpl  -> ebeamXTCAVAmpl ;  /* XTCAV Amplitude in MVolt. */
  double _fEbeamXTCAVPhase -> ebeamXTCAVPhase;  /* XTCAV Phase in degrees. */
  double _fEbeamDumpCharge -> ebeamDumpCharge;  /* Bunch charge at Dump in num. electrons */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV6 ------------------
@type BldDataEBeamV6
  [[type_id(Id_EBeam, 6)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
    EbeamPkCurrBC1Damage = 0x100,
    EbeamEnergyBC1Damage = 0x200,
    EbeamUndPosXDamage = 0x400,
    EbeamUndPosYDamage = 0x800,
    EbeamUndAngXDamage = 0x1000,
    EbeamUndAngYDamage = 0x2000,
    EbeamXTCAVAmplDamage  = 0x4000,
    EbeamXTCAVPhaseDamage = 0x8000,
    EbeamDumpChargeDamage = 0x10000,
    EbeamPhotonEnergyDamage = 0x20000
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;  /* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;  /* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;  /* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;  /* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;  /* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;  /* Undulator launch feedback beam y-angle in mrad. */
  double _fEbeamXTCAVAmpl  -> ebeamXTCAVAmpl ;  /* XTCAV Amplitude in MVolt. */
  double _fEbeamXTCAVPhase -> ebeamXTCAVPhase;  /* XTCAV Phase in degrees. */
  double _fEbeamDumpCharge -> ebeamDumpCharge;  /* Bunch charge at Dump in num. electrons */
  double _fEbeamPhotonEnergy  -> ebeamPhotonEnergy ;  /* computed photon energy, in eV */
  double _fEbeamLTU250 -> ebeamLTU250;  /* LTU250 BPM value in mm, used to compute photon energy. from BPMS:LTU1:250:X */
  double _fEbeamLTU450 -> ebeamLTU450;  /* LTU450 BPM value in mm, used to compute photon energy. from BPMS:LTU1:450:X */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataEBeamV7 ------------------
/* BldDataEBeamV7 is the same as BldDataEBeamV6. */
/* A sign-error error was discovered in the calculation of the photon energy that goes into the ebeam bld.  */
/* This is fixed on the accelerator side, but we will increment the ebeam bld version number to V7 so the   */
/* data is clearly marked as changed. */
@type BldDataEBeamV7
  [[type_id(Id_EBeam, 7)]]
  [[value_type]]
  [[pack(4)]]
{
  /* Constants defining bit mask for individual damage bits in value returned from damageMask() */
  @enum DamageMask (int32_t) {
    EbeamChargeDamage = 0x001,
    EbeamL3EnergyDamage = 0x002,
    EbeamLTUPosXDamage = 0x004,
    EbeamLTUPosYDamage = 0x008,
    EbeamLTUAngXDamage = 0x010,
    EbeamLTUAngYDamage = 0x020,
    EbeamPkCurrBC2Damage = 0x040,
    EbeamEnergyBC2Damage = 0x080,
    EbeamPkCurrBC1Damage = 0x100,
    EbeamEnergyBC1Damage = 0x200,
    EbeamUndPosXDamage = 0x400,
    EbeamUndPosYDamage = 0x800,
    EbeamUndAngXDamage = 0x1000,
    EbeamUndAngYDamage = 0x2000,
    EbeamXTCAVAmplDamage  = 0x4000,
    EbeamXTCAVPhaseDamage = 0x8000,
    EbeamDumpChargeDamage = 0x10000,
    EbeamPhotonEnergyDamage = 0x20000
  }

  uint32_t _uDamageMask -> damageMask;  /* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;  /* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;  /* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;  /* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;  /* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;  /* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;  /* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;  /* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;  /* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;  /* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;  /* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;  /* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;  /* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;  /* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;  /* Undulator launch feedback beam y-angle in mrad. */
  double _fEbeamXTCAVAmpl  -> ebeamXTCAVAmpl ;  /* XTCAV Amplitude in MVolt. */
  double _fEbeamXTCAVPhase -> ebeamXTCAVPhase;  /* XTCAV Phase in degrees. */
  double _fEbeamDumpCharge -> ebeamDumpCharge;  /* Bunch charge at Dump in num. electrons */
  double _fEbeamPhotonEnergy  -> ebeamPhotonEnergy ;  /* computed photon energy, in eV */
  double _fEbeamLTU250 -> ebeamLTU250;  /* LTU250 BPM value in mm, used to compute photon energy. from BPMS:LTU1:250:X */
  double _fEbeamLTU450 -> ebeamLTU450;  /* LTU450 BPM value in mm, used to compute photon energy. from BPMS:LTU1:450:X */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataPhaseCavity ------------------
/* PV names: UND:R02:IOC:16:BAT:FitTime1, UND:R02:IOC:16:BAT:FitTime2,
                UND:R02:IOC:16:BAT:Charge1,  UND:R02:IOC:16:BAT:Charge2 */
@type BldDataPhaseCavity
  [[type_id(Id_PhaseCavity, 0)]]
  [[value_type]]
  [[pack(4)]]
{
  double _fFitTime1 -> fitTime1;  /* UND:R02:IOC:16:BAT:FitTime1 value in pico-seconds. */
  double _fFitTime2 -> fitTime2;  /* UND:R02:IOC:16:BAT:FitTime2 value in pico-seconds. */
  double _fCharge1 -> charge1;  /* UND:R02:IOC:16:BAT:Charge1 value in pico-columbs. */
  double _fCharge2 -> charge2;  /* UND:R02:IOC:16:BAT:Charge2 value in pico-columbs. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataIpimbV0 ------------------
/* Combined structure which includes Ipimb.DataV1, Ipimb.ConfigV1, and
            Lusi.IpmFexV1 objects. */
@type BldDataIpimbV0
  [[type_id(Id_SharedIpimb, 0)]]
  [[pack(4)]]
{
  Ipimb.DataV1 _ipimbData -> ipimbData;
  Ipimb.ConfigV1 _ipimbConfig -> ipimbConfig;
  Lusi.IpmFexV1 _ipmFexData -> ipmFexData;
}


//------------------ BldDataIpimbV1 ------------------
/* Combined structure which includes Ipimb.DataV2, Ipimb.ConfigV2, and
            Lusi.IpmFexV1 objects. */
@type BldDataIpimbV1
  [[type_id(Id_SharedIpimb, 1)]]
  [[pack(4)]]
{
  Ipimb.DataV2 _ipimbData -> ipimbData;
  Ipimb.ConfigV2 _ipimbConfig -> ipimbConfig;
  Lusi.IpmFexV1 _ipmFexData -> ipmFexData;
}


//------------------ BldDataPimV1 ------------------
/* Combined structure which includes Pulnix.TM6740ConfigV2, Lusi.PimImageConfigV1, and
            Camera.FrameV1 objects. */
@type BldDataPimV1
  [[type_id(Id_SharedPim, 1)]]
  [[pack(4)]]
{
  Pulnix.TM6740ConfigV2 _camConfig -> camConfig;
  Lusi.PimImageConfigV1 _pimConfig -> pimConfig;
  Camera.FrameV1 _frame -> frame;
}


//------------------ BldDataGMDV0 ------------------
/* Gas Monitor Detector data. */
@type BldDataGMDV0
  [[type_id(Id_GMD, 0)]]
  [[pack(4)]]
{
  char _strGasType[32] -> gasType  [[shape_method(None)]];  /* String describing gas type */
  double _fPressure -> pressure;  /* Pressure from Spinning Rotor Gauge */
  double _fTemperature -> temperature;  /* Temp from PT100 */
  double _fCurrent -> current;  /* Current from Keithley Electrometer */
  double _fHvMeshElectron -> hvMeshElectron;  /* HV Mesh Electron */
  double _fHvMeshIon -> hvMeshIon;  /* HV Mesh Ion */
  double _fHvMultIon -> hvMultIon;  /* HV Mult Ion */
  double _fChargeQ -> chargeQ;  /* Charge Q */
  double _fPhotonEnergy -> photonEnergy;  /* Photon Energy */
  double _fMultPulseIntensity -> multPulseIntensity;  /* Pulse Intensity derived from Electron Multiplier */
  double _fKeithleyPulseIntensity -> keithleyPulseIntensity;  /* Pulse Intensity derived from ION cup current */
  double _fPulseEnergy -> pulseEnergy;  /* Pulse Energy derived from Electron Multiplier */
  double _fPulseEnergyFEE -> pulseEnergyFEE;  /* Pulse Energy from FEE Gas Detector */
  double _fTransmission -> transmission;  /* Transmission derived from Electron Multiplier */
  double _fTransmissionFEE -> transmissionFEE;  /* Transmission from FEE Gas Detector */
  double _fSpare6;
}


//------------------ BldDataGMDV1 ------------------
/* Gas Monitor Detector data. */
@type BldDataGMDV1
  [[type_id(Id_GMD, 1)]]
  [[pack(4)]]
{
  double _fMilliJoulesPerPulse -> milliJoulesPerPulse;  /* Shot to shot pulse energy (mJ) */
  double _fMilliJoulesAverage -> milliJoulesAverage;  /* Average pulse energy from ION cup current (mJ) */
  double _fCorrectedSumPerPulse -> correctedSumPerPulse;  /* Bg corrected waveform integrated within limits in raw A/D counts */
  double _fBgValuePerSample -> bgValuePerSample;  /* Avg background value per sample in raw A/D counts */
  double _fRelativeEnergyPerPulse -> relativeEnergyPerPulse;  /* Shot by shot pulse energy in arbitrary units */
  double _fSpare1;
}

//------------------ BldDataGMDV2 ------------------
/* Gas Monitor Detector data. */
@type BldDataGMDV2
  [[type_id(Id_GMD, 2)]]
  [[pack(4)]]
{
  double _fMilliJoulesPerPulse    -> milliJoulesPerPulse;     /* Shot to shot pulse energy (mJ).  Not as robust as relativeEnergyPerPulse() method. */
  double _fMilliJoulesAverage     -> milliJoulesAverage;      /* Average pulse energy from ION cup current (mJ).  Not as robust as relativeEnergyPerPulse() method. */
  double _fSumAllPeaksFiltBkgd    -> sumAllPeaksFiltBkgd;     /* Sum of all peaks, normalized w/ filt bkgd level.  Not typically used by the user. */
  double _fRawAvgBkgd             -> rawAvgBkgd;              /* Avg background value per waveform in raw A/D counts.  Not typically used by the user. */
  double _fRelativeEnergyPerPulse -> relativeEnergyPerPulse;  /* Shot by shot pulse energy in arbitrary units.  The most stable measurement.  Most users should use this. */
  double _fSumAllPeaksRawBkgd     -> sumAllPeaksRawBkgd;      /* Sum of all peaks, normalized w/ raw avg bkgd level.  Not typically used by the user. */
}

//------------------ BldDataAcqADCV1 ------------------
/* Combined structure which includes Acqiris.ConfigV1 and
            Acqiris.DataDescV1 objects. */
@type BldDataAcqADCV1
  [[type_id(Id_SharedAcqADC, 1)]]
  [[no_sizeof]]
  [[pack(4)]]
  [[config(Acqiris.ConfigV1)]]
{
  Acqiris.ConfigV1 _config -> config;
  Acqiris.DataDescV1 _data -> data;
}


//------------------ BldDataSpectrometerV0 ------------------
/* Structure which contains image projections for spectrometers. */
@type BldDataSpectrometerV0
  [[type_id(Id_Spectrometer, 0)]]
  [[pack(4)]]
{
  uint32_t _hproj[1024] -> hproj;
  uint32_t _vproj[256] -> vproj;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ BldDataSpectrometerV1 ------------------
/* Structure which contains image projections and fit parameters for spectrometers. 
	Changes from V0 include extending size of hproj, removal of vproj,
	 and addition of fit parameters. */
@type BldDataSpectrometerV1
  [[type_id(Id_Spectrometer, 1)]]
  [[pack(4)]]
{
  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Construct from dimensions only.  Allow data to be appended externally. */
  @init(width -> _width, nPeaks -> _nPeaks)  [[inline]];

  /* Width of camera frame and thus size of hproj array 
     PV TBD */
  uint32_t _width -> width; 

  /* First row of pixels used in projection ROI  
     PV TBD */
  uint32_t _hproj_y1 -> hproj_y1;

  /* Last row of pixels used in projection ROI
    PV: TBD */
  uint32_t _hproj_y2 -> hproj_y2;

  /* Raw center of mass, no baseline subtraction 
     PV: TBD */
  double _fComRaw -> comRaw;

  /* Baseline level for calculated values 
     PV: TBD */
  double _fBaseline -> baseline;

  /* Baseline-subtracted center of mass 
     PV: TBD */
  double _fCom -> com;

  /* Integrated area under spectrum (no baseline subtraction) 
     PV: TBD */
  double _fIntegral -> integral;

  /* Number of peak fits performed
    PV: TBD */ 
  uint32_t _nPeaks -> nPeaks;

  /* Projection of spectrum onto energy axis 
     PV TBD */
  int32_t _hproj[@self._width] -> hproj; 

  /* Peak position array, length given by nPeaks
     PV: TBD */
  double _peakPos[@self._nPeaks] -> peakPos;

  /* Peak height array, length given by nPeaks
     PV: TBD */
  double _peakHeight[@self._nPeaks] -> peakHeight;

  /* Peak FWHM array, length given by nPeaks
     PV: TBD */
  double _Fwhm[@self._nPeaks] -> FWHM;

}

//------------------ BldDataAnalogInputV1 ------------------
/* Structure which contains voltage data from an analog input device. */
@type BldDataAnalogInputV1
  [[type_id(Id_AnalogInput, 1)]]
  [[pack(4)]]
{
  /* Constructor which takes values for every attribute. */
  @init()  [[auto, inline]];

  /* The number of active channels on the analog input device. */
  uint32_t _numChannels -> numChannels;

  /* Array of voltage values were each entry represents a channel of the analog input device. */
  double _channelVoltages[@self._numChannels] -> channelVoltages;
}

} //- @package Bld
