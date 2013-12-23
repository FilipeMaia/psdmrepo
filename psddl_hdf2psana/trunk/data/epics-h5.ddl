@include "psddldata/epics.ddl";
@package Epics  {


//------------------ epicsTimeStamp ------------------
@h5schema epicsTimeStamp
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute secPastEpoch [[method(sec)]];
    @attribute nsec;
  }
}


//------------------ dbr_time_string ------------------
@h5schema dbr_time_string
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_short ------------------
@h5schema dbr_time_short
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_float ------------------
@h5schema dbr_time_float
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_enum ------------------
@h5schema dbr_time_enum
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_char ------------------
@h5schema dbr_time_char
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_long ------------------
@h5schema dbr_time_long
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_time_double ------------------
@h5schema dbr_time_double
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_sts_string ------------------
@h5schema dbr_sts_string
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_short ------------------
@h5schema dbr_ctrl_short
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_float ------------------
@h5schema dbr_ctrl_float
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_enum ------------------
@h5schema dbr_ctrl_enum
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_char ------------------
@h5schema dbr_ctrl_char
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_long ------------------
@h5schema dbr_ctrl_long
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ dbr_ctrl_double ------------------
@h5schema dbr_ctrl_double
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ EpicsPvHeader ------------------
@h5schema EpicsPvHeader
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pvId;
    @attribute dbrType;
    @attribute numElements;
  }
}


//------------------ EpicsPvCtrlHeader ------------------
@h5schema EpicsPvCtrlHeader
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeHeader ------------------
@h5schema EpicsPvTimeHeader
  [[version(0)]]
  [[external]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlString ------------------
@h5schema EpicsPvCtrlString
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlShort ------------------
@h5schema EpicsPvCtrlShort
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlFloat ------------------
@h5schema EpicsPvCtrlFloat
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlEnum ------------------
@h5schema EpicsPvCtrlEnum
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlChar ------------------
@h5schema EpicsPvCtrlChar
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlLong ------------------
@h5schema EpicsPvCtrlLong
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvCtrlDouble ------------------
@h5schema EpicsPvCtrlDouble
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeString ------------------
@h5schema EpicsPvTimeString
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeShort ------------------
@h5schema EpicsPvTimeShort
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeFloat ------------------
@h5schema EpicsPvTimeFloat
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeEnum ------------------
@h5schema EpicsPvTimeEnum
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeChar ------------------
@h5schema EpicsPvTimeChar
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeLong ------------------
@h5schema EpicsPvTimeLong
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ EpicsPvTimeDouble ------------------
@h5schema EpicsPvTimeDouble
  [[version(0)]]
  [[external("psddl_hdf2psana/epics.h")]]
  [[embedded]]
{
}


//------------------ PvConfigV1 ------------------
@h5schema PvConfigV1
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pvId;
    @attribute description;
    @attribute interval;
  }
}


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute numPv;
  }
  @dataset pvConfig [[method(getPvConfig)]];
}
} //- @package Epics
