@include "psddldata/evr.ddl";
@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/DetInfo.hh")]];
@package EvrData  {


//------------------ PulseConfig ------------------
@h5schema PulseConfig
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ PulseConfigV3 ------------------
@h5schema PulseConfigV3
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ EventCodeV3 ------------------
@h5schema EventCodeV3
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ EventCodeV4 ------------------
@h5schema EventCodeV4
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ EventCodeV5 ------------------
@h5schema EventCodeV5
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ EventCodeV6 ------------------
@h5schema EventCodeV6
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ OutputMap ------------------
@h5schema OutputMap
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute source;
    @attribute source_id;
    @attribute conn;
    @attribute conn_id;
  }
}


//------------------ OutputMapV2 ------------------
@h5schema OutputMapV2
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute source;
    @attribute source_id;
    @attribute conn;
    @attribute conn_id;
    @attribute module;
  }
}


//------------------ SequencerEntry ------------------
@h5schema SequencerEntry
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ SequencerConfigV1 ------------------
@h5schema SequencerConfigV1
  [[version(0)]]
  [[embedded]]
{
  @dataset config {
    @attribute sync_source;
    @attribute beam_source;
    @attribute length;
    @attribute cycles;
    @attribute entries [[vlen]];
  }
}


//------------------ ConfigV5 ------------------
@h5schema ConfigV5
  [[version(0)]]
{
  @dataset config {
    @attribute neventcodes;
    @attribute npulses;
    @attribute noutputs;
  }
  @dataset eventcodes;
  @dataset pulses;
  @dataset output_maps;
  @dataset seq_config;
}


//------------------ ConfigV6 ------------------
@h5schema ConfigV6
  [[version(0)]]
{
  @dataset config {
    @attribute neventcodes;
    @attribute npulses;
    @attribute noutputs;
  }
  @dataset eventcodes;
  @dataset pulses;
  @dataset output_maps;
  @dataset seq_config;
}


//------------------ ConfigV7 ------------------
@h5schema ConfigV7
  [[version(0)]]
{
  @dataset config {
    @attribute neventcodes;
    @attribute npulses;
    @attribute noutputs;
  }
  @dataset eventcodes;
  @dataset pulses;
  @dataset output_maps;
  @dataset seq_config;
}


//------------------ FIFOEvent ------------------
@h5schema FIFOEvent
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ DataV3 ------------------
@h5schema DataV3
  [[version(0)]]
  [[external("psddl_hdf2psana/evr.h")]]
{
}

//------------------ DataV4 ------------------
@h5schema DataV4
  [[version(0)]]
  [[external("psddl_hdf2psana/evr.h")]]
{
}


//------------------ IOChannel ------------------
@h5schema IOChannel
  [[version(0)]]
  [[external("psddl_hdf2psana/evr.h")]]
  [[embedded]]
{
}


//------------------ IOConfigV1 ------------------
@h5schema IOConfigV1
  [[version(0)]]
  [[external("psddl_hdf2psana/evr.h")]]
{
}

//------------------ IOChannelV2 ------------------
@h5schema IOChannelV2
  [[version(0)]]
  [[embedded]]
  [[external("psddl_hdf2psana/evr.h")]]
{
}

//------------------ IOConfigV2 ------------------
@h5schema IOConfigV2
  [[version(0)]]
  [[external("psddl_hdf2psana/evr.h")]]
{
}

} //- @package EvrData
