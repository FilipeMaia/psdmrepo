@include "psddldata/acqiris.ddl";
@package Acqiris  {


//------------------ VertV1 ------------------
@h5schema VertV1
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute fullScale;
    @attribute offset;
    @attribute coupling;
    @attribute bandwidth;
  }
}


//------------------ HorizV1 ------------------
@h5schema HorizV1
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ TrigV1 ------------------
@h5schema TrigV1
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute nbrConvertersPerChannel;
    @attribute channelMask;
    @attribute nbrBanks;
    @attribute nbrChannels;
  }
  @dataset horiz;
  @dataset trig;
  @dataset vert;
}


//------------------ TimestampV1 ------------------
@h5schema TimestampV1
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pos;
    @attribute value;
  }
}


//------------------ DataDescV1Elem ------------------
@h5schema DataDescV1Elem
  [[version(0)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
  [[embedded]]
{
}


//------------------ DataDescV1Elem ------------------
@h5schema DataDescV1Elem
  [[version(1)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
  [[embedded]]
{
}


//------------------ DataDescV1 ------------------
@h5schema DataDescV1
  [[version(0)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
{
}


//------------------ DataDescV1 ------------------
@h5schema DataDescV1
  [[version(1)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
{
}


//------------------ TdcChannel ------------------
@h5schema TdcChannel
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ TdcAuxIO ------------------
@h5schema TdcAuxIO
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ TdcVetoIO ------------------
@h5schema TdcVetoIO
  [[version(0)]]
  [[embedded]]
  [[default]]
{
}


//------------------ TdcConfigV1 ------------------
@h5schema TdcConfigV1
  [[version(0)]]
{
  @dataset veto;
  @dataset channel [[method(channels)]];
  @dataset auxio;
}


//------------------ TdcDataV1_Item ------------------
@h5schema TdcDataV1_Item
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute source;
    @attribute overflow [[method(bf_ofv_)]];
    @attribute value [[method(bf_val_)]];
  }
}


//------------------ TdcDataV1Common ------------------
@h5schema TdcDataV1Common
  [[version(0)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
  [[embedded]]
{
}


//------------------ TdcDataV1Channel ------------------
@h5schema TdcDataV1Channel
  [[version(0)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
  [[embedded]]
{
}


//------------------ TdcDataV1Marker ------------------
@h5schema TdcDataV1Marker
  [[version(0)]]
  [[external("psddl_hdf2psana/acqiris.h")]]
  [[embedded]]
{
}


//------------------ TdcDataV1 ------------------
@h5schema TdcDataV1
  [[version(0)]]
{
  @dataset data [[vlen]];
}
} //- @package Acqiris
