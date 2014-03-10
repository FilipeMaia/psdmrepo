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
  double _f_11_ENRC -> f_11_ENRC;	/* Value of GDET:FEE1:11:ENRC, in mJ. */
  double _f_12_ENRC -> f_12_ENRC;	/* Value of GDET:FEE1:12:ENRC, in mJ. */
  double _f_21_ENRC -> f_21_ENRC;	/* Value of GDET:FEE1:21:ENRC, in mJ. */
  double _f_22_ENRC -> f_22_ENRC;	/* Value of GDET:FEE1:22:ENRC, in mJ. */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;	/* Beam current in Amps. */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;	/* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;	/* Beam energy in MeV. */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;	/* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;	/* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;	/* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;	/* Beam position in mm (related to beam energy). */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;	/* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;	/* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;	/* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;	/* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;	/* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;	/* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;	/* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;	/* Undulator launch feedback beam y-angle in mrad. */

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

  uint32_t _uDamageMask -> damageMask;	/* Damage mask. */
  double _fEbeamCharge -> ebeamCharge;	/* Beam charge in nC. */
  double _fEbeamL3Energy -> ebeamL3Energy;	/* Beam energy in MeV. */
  double _fEbeamLTUPosX -> ebeamLTUPosX;	/* LTU beam position (BPMS:LTU1:720 through 750) in mm. */
  double _fEbeamLTUPosY -> ebeamLTUPosY;	/* LTU beam position in mm. */
  double _fEbeamLTUAngX -> ebeamLTUAngX;	/* LTU beam angle in mrad. */
  double _fEbeamLTUAngY -> ebeamLTUAngY;	/* LTU beam angle in mrad. */
  double _fEbeamPkCurrBC2 -> ebeamPkCurrBC2;	/* Beam current in Amps. */
  double _fEbeamEnergyBC2 -> ebeamEnergyBC2;	/* Beam position in mm (related to beam energy). */
  double _fEbeamPkCurrBC1 -> ebeamPkCurrBC1;	/* Beam current in Amps. */
  double _fEbeamEnergyBC1 -> ebeamEnergyBC1;	/* Beam position in mm (related to beam energy). */
  double _fEbeamUndPosX -> ebeamUndPosX;	/* Undulator launch feedback (BPMs U4 through U10) beam x-position in mm. */
  double _fEbeamUndPosY -> ebeamUndPosY;	/* Undulator launch feedback beam y-position in mm. */
  double _fEbeamUndAngX -> ebeamUndAngX;	/* Undulator launch feedback beam x-angle in mrad. */
  double _fEbeamUndAngY -> ebeamUndAngY;	/* Undulator launch feedback beam y-angle in mrad. */
  double _fEbeamXTCAVAmpl  -> ebeamXTCAVAmpl ;	/* XTCAV Amplitude in MVolt. */
  double _fEbeamXTCAVPhase -> ebeamXTCAVPhase;	/* XTCAV Phase in degrees. */
  double _fEbeamDumpCharge -> ebeamDumpCharge;	/* Bunch charge at Dump in num. electrons */

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
  double _fFitTime1 -> fitTime1;	/* UND:R02:IOC:16:BAT:FitTime1 value in pico-seconds. */
  double _fFitTime2 -> fitTime2;	/* UND:R02:IOC:16:BAT:FitTime2 value in pico-seconds. */
  double _fCharge1 -> charge1;	/* UND:R02:IOC:16:BAT:Charge1 value in pico-columbs. */
  double _fCharge2 -> charge2;	/* UND:R02:IOC:16:BAT:Charge2 value in pico-columbs. */

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
  char _strGasType[32] -> gasType  [[shape_method(None)]];	/* String describing gas type */
  double _fPressure -> pressure;	/* Pressure from Spinning Rotor Gauge */
  double _fTemperature -> temperature;	/* Temp from PT100 */
  double _fCurrent -> current;	/* Current from Keithley Electrometer */
  double _fHvMeshElectron -> hvMeshElectron;	/* HV Mesh Electron */
  double _fHvMeshIon -> hvMeshIon;	/* HV Mesh Ion */
  double _fHvMultIon -> hvMultIon;	/* HV Mult Ion */
  double _fChargeQ -> chargeQ;	/* Charge Q */
  double _fPhotonEnergy -> photonEnergy;	/* Photon Energy */
  double _fMultPulseIntensity -> multPulseIntensity;	/* Pulse Intensity derived from Electron Multiplier */
  double _fKeithleyPulseIntensity -> keithleyPulseIntensity;	/* Pulse Intensity derived from ION cup current */
  double _fPulseEnergy -> pulseEnergy;	/* Pulse Energy derived from Electron Multiplier */
  double _fPulseEnergyFEE -> pulseEnergyFEE;	/* Pulse Energy from FEE Gas Detector */
  double _fTransmission -> transmission;	/* Transmission derived from Electron Multiplier */
  double _fTransmissionFEE -> transmissionFEE;	/* Transmission from FEE Gas Detector */
  double _fSpare6;
}


//------------------ BldDataGMDV1 ------------------
/* Gas Monitor Detector data. */
@type BldDataGMDV1
  [[type_id(Id_GMD, 1)]]
  [[pack(4)]]
{
  double _fMilliJoulesPerPulse -> milliJoulesPerPulse;	/* Shot to shot pulse energy (mJ) */
  double _fMilliJoulesAverage -> milliJoulesAverage;	/* Average pulse energy from ION cup current (mJ) */
  double _fCorrectedSumPerPulse -> correctedSumPerPulse;	/* Bg corrected waveform integrated within limits in raw A/D counts */
  double _fBgValuePerSample -> bgValuePerSample;	/* Avg background value per sample in raw A/D counts */
  double _fRelativeEnergyPerPulse -> relativeEnergyPerPulse;	/* Shot by shot pulse energy in arbitrary units */
  double _fSpare1;
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
} //- @package Bld
