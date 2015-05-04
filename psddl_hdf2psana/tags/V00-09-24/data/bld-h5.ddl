@include "psddldata/bld.ddl";
@include "psddldata/acqiris.ddl";
@include "psddldata/camera.ddl";
@include "psddldata/ipimb.ddl";
@include "psddldata/lusi.ddl";
@include "psddldata/pulnix.ddl";
@include "psddl_hdf2psana/lusi-h5.ddl" [[headers("psddl_hdf2psana/lusi.ddl.h")]];
@include "psddl_hdf2psana/ipimb-h5.ddl" [[headers("psddl_hdf2psana/ipimb.ddl.h")]];
@package Bld  {


//------------------ BldDataEBeamV0 ------------------
@h5schema BldDataEBeamV0
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
  }
}


//------------------ BldDataEBeamV1 ------------------
@h5schema BldDataEBeamV1
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
  }
}


//------------------ BldDataEBeamV2 ------------------
@h5schema BldDataEBeamV2
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
  }
}


//------------------ BldDataEBeamV3 ------------------
@h5schema BldDataEBeamV3
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
    @attribute fEbeamPkCurrBC1 [[method(ebeamPkCurrBC1)]];
    @attribute fEbeamEnergyBC1 [[method(ebeamEnergyBC1)]];
  }
}


//------------------ BldDataEBeamV4 ------------------
@h5schema BldDataEBeamV4
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
    @attribute fEbeamPkCurrBC1 [[method(ebeamPkCurrBC1)]];
    @attribute fEbeamEnergyBC1 [[method(ebeamEnergyBC1)]];
    @attribute fEbeamUndPosX [[method(ebeamUndPosX)]];
    @attribute fEbeamUndPosY [[method(ebeamUndPosY)]];
    @attribute fEbeamUndAngX [[method(ebeamUndAngX)]];
    @attribute fEbeamUndAngY [[method(ebeamUndAngY)]];
  }
}

//------------------ BldDataEBeamV5 ------------------
@h5schema BldDataEBeamV5
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
    @attribute fEbeamPkCurrBC1 [[method(ebeamPkCurrBC1)]];
    @attribute fEbeamEnergyBC1 [[method(ebeamEnergyBC1)]];
    @attribute fEbeamUndPosX [[method(ebeamUndPosX)]];
    @attribute fEbeamUndPosY [[method(ebeamUndPosY)]];
    @attribute fEbeamUndAngX [[method(ebeamUndAngX)]];
    @attribute fEbeamUndAngY [[method(ebeamUndAngY)]];
    @attribute fEbeamXTCAVAmpl [[method(ebeamXTCAVAmpl)]];
    @attribute fEbeamXTCAVPhase [[method(ebeamXTCAVPhase)]];
    @attribute fEbeamDumpCharge [[method(ebeamDumpCharge)]];
  }
}

//------------------ BldDataEBeamV6 ------------------
@h5schema BldDataEBeamV6
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
    @attribute fEbeamPkCurrBC1 [[method(ebeamPkCurrBC1)]];
    @attribute fEbeamEnergyBC1 [[method(ebeamEnergyBC1)]];
    @attribute fEbeamUndPosX [[method(ebeamUndPosX)]];
    @attribute fEbeamUndPosY [[method(ebeamUndPosY)]];
    @attribute fEbeamUndAngX [[method(ebeamUndAngX)]];
    @attribute fEbeamUndAngY [[method(ebeamUndAngY)]];
    @attribute fEbeamXTCAVAmpl [[method(ebeamXTCAVAmpl)]];
    @attribute fEbeamXTCAVPhase [[method(ebeamXTCAVPhase)]];
    @attribute fEbeamDumpCharge [[method(ebeamDumpCharge)]];
    @attribute fEbeamPhotonEnergy  [[method(ebeamPhotonEnergy)]];
    @attribute fEbeamLTU250 [[method(ebeamLTU250)]];
    @attribute fEbeamLTU450 [[method(ebeamLTU450)]];
  }
}


//------------------ BldDataEBeamV7 ------------------
@h5schema BldDataEBeamV7
  [[version(0)]]
{
  @dataset data {
    @attribute uDamageMask [[method(damageMask)]];
    @attribute fEbeamCharge [[method(ebeamCharge)]];
    @attribute fEbeamL3Energy [[method(ebeamL3Energy)]];
    @attribute fEbeamLTUPosX [[method(ebeamLTUPosX)]];
    @attribute fEbeamLTUPosY [[method(ebeamLTUPosY)]];
    @attribute fEbeamLTUAngX [[method(ebeamLTUAngX)]];
    @attribute fEbeamLTUAngY [[method(ebeamLTUAngY)]];
    @attribute fEbeamPkCurrBC2 [[method(ebeamPkCurrBC2)]];
    @attribute fEbeamEnergyBC2 [[method(ebeamEnergyBC2)]];
    @attribute fEbeamPkCurrBC1 [[method(ebeamPkCurrBC1)]];
    @attribute fEbeamEnergyBC1 [[method(ebeamEnergyBC1)]];
    @attribute fEbeamUndPosX [[method(ebeamUndPosX)]];
    @attribute fEbeamUndPosY [[method(ebeamUndPosY)]];
    @attribute fEbeamUndAngX [[method(ebeamUndAngX)]];
    @attribute fEbeamUndAngY [[method(ebeamUndAngY)]];
    @attribute fEbeamXTCAVAmpl [[method(ebeamXTCAVAmpl)]];
    @attribute fEbeamXTCAVPhase [[method(ebeamXTCAVPhase)]];
    @attribute fEbeamDumpCharge [[method(ebeamDumpCharge)]];
    @attribute fEbeamPhotonEnergy  [[method(ebeamPhotonEnergy)]];
    @attribute fEbeamLTU250 [[method(ebeamLTU250)]];
    @attribute fEbeamLTU450 [[method(ebeamLTU450)]];
  }
}


//------------------ BldDataPhaseCavity ------------------
@h5schema BldDataPhaseCavity
  [[version(0)]]
{
  @dataset data {
    @attribute fFitTime1 [[method(fitTime1)]];
    @attribute fFitTime2 [[method(fitTime2)]];
    @attribute fCharge1 [[method(charge1)]];
    @attribute fCharge2 [[method(charge2)]];
  }
}


//------------------ BldDataIpimbV0 ------------------
@h5schema BldDataIpimbV0
  [[version(0)]]
{
  @dataset data {
    @attribute ipimbData;
    @attribute ipimbConfig;
    @attribute ipmFexData;
  }
}


//------------------ BldDataIpimbV1 ------------------
@h5schema BldDataIpimbV1
  [[version(0)]]
{
  @dataset data {
    @attribute ipimbData;
    @attribute ipimbConfig;
    @attribute ipmFexData;
  }
}


//------------------ BldDataPimV1 ------------------
@h5schema BldDataPimV1
  [[version(0)]]
  [[external("psddl_hdf2psana/bld.h")]]
{
}


//------------------ BldDataGMDV0 ------------------
@h5schema BldDataGMDV0
  [[version(0)]]
{
  @dataset data {
    @attribute strGasType [[method(gasType)]];
    @attribute fPressure [[method(pressure)]];
    @attribute fTemperature [[method(temperature)]];
    @attribute fCurrent [[method(current)]];
    @attribute fHvMeshElectron [[method(hvMeshElectron)]];
    @attribute fHvMeshIon [[method(hvMeshIon)]];
    @attribute fHvMultIon [[method(hvMultIon)]];
    @attribute fChargeQ [[method(chargeQ)]];
    @attribute fPhotonEnergy [[method(photonEnergy)]];
    @attribute fMultPulseIntensity [[method(multPulseIntensity)]];
    @attribute fKeithleyPulseIntensity [[method(keithleyPulseIntensity)]];
    @attribute fPulseEnergy [[method(pulseEnergy)]];
    @attribute fPulseEnergyFEE [[method(pulseEnergyFEE)]];
    @attribute fTransmission [[method(transmission)]];
    @attribute fTransmissionFEE [[method(transmissionFEE)]];
  }
}


//------------------ BldDataGMDV1 ------------------
@h5schema BldDataGMDV1
  [[version(0)]]
{
  @dataset data {
    @attribute fMilliJoulesPerPulse [[method(milliJoulesPerPulse)]];
    @attribute fMilliJoulesAverage [[method(milliJoulesAverage)]];
    @attribute fCorrectedSumPerPulse [[method(correctedSumPerPulse)]];
    @attribute fBgValuePerSample [[method(bgValuePerSample)]];
    @attribute fRelativeEnergyPerPulse [[method(relativeEnergyPerPulse)]];
  }
}

//------------------ BldDataGMDV2 ------------------
@h5schema BldDataGMDV2
  [[version(0)]]
{
  @dataset data {
    @attribute fMilliJoulesPerPulse [[method(milliJoulesPerPulse)]];
    @attribute fMilliJoulesAverage [[method(milliJoulesAverage)]];
    @attribute fSumAllPeaksFiltBkgd [[method(sumAllPeaksFiltBkgd)]];
    @attribute fRawAvgBkgd [[method(rawAvgBkgd)]];
    @attribute fRelativeEnergyPerPulse [[method(relativeEnergyPerPulse)]];
    @attribute fSumAllPeaksRawBkgd [[method(sumAllPeaksRawBkgd)]];
  }
}


//------------------ BldDataAcqADCV1 ------------------
@h5schema BldDataAcqADCV1
  [[version(0)]]
  [[external("psddl_hdf2psana/bld.h")]]
{
}


//------------------ BldDataSpectrometerV0 ------------------
@h5schema BldDataSpectrometerV0
  [[version(0)]]
{
  @dataset data {
    @attribute hproj;
    @attribute vproj;
  }
}

//------------------ BldDataSpectrometerV1 ------------------
@h5schema BldDataSpectrometerV1
  [[version(0)]]
{
  @dataset data {
    @attribute width;
    @attribute hproj_y1;
    @attribute hproj_y2;
    @attribute comRaw;
    @attribute baseline;
    @attribute com;
    @attribute integral;
    @attribute nPeaks;
  }
  @dataset hproj;
  @dataset peakPos [[zero_dims]];
  @dataset peakHeight [[zero_dims]];
  @dataset FWHM [[zero_dims]];
}

//------------------ BldDataAnalogInputV1 ------------------
/* Structure which contains voltage data from an analog input device. */
@h5schema BldDataAnalogInputV1
  [[version(0)]]
{
  @dataset data {
    @attribute numChannels;
  }
  @dataset channelVoltages [[zero_dims]];
}


} //- @package Bld
