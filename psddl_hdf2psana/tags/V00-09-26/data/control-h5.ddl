@include "psddldata/control.ddl";
@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/ClockTime.hh")]];
@include "psddl_hdf2psana/xtc-h5.ddl" [[headers("psddl_hdf2psana/xtc.h")]];
@package ControlData  {


//------------------ PVControl ------------------
@h5schema PVControl
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute name;
    @attribute int32_t index;
    @attribute value;
  }
}


//------------------ PVMonitor ------------------
@h5schema PVMonitor
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute name;
    @attribute int32_t index;
    @attribute loValue;
    @attribute hiValue;
  }
}


//------------------ PVLabel ------------------
@h5schema PVLabel
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute name;
    @attribute value;
  }
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(0)]]
{
  @dataset config {
    @attribute events;
    @attribute uses_duration;
    @attribute uses_events;
    @attribute duration;
    @attribute npvControls;
    @attribute npvMonitors;
  }
  @dataset pvControls;
  @dataset pvMonitors;
  @dataset pvLabels;
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(1)]]
{
  @dataset config {
    @attribute events;
    @attribute uses_duration;
    @attribute uses_events;
    @attribute duration;
    @attribute npvControls;
    @attribute npvLabels;
  }
  @dataset pvControls;
  @dataset pvMonitors;
  @dataset pvLabels;
}


//------------------ ConfigV2 ------------------
@h5schema ConfigV2
  [[version(2)]]
{
  @dataset config {
    @attribute events;
    @attribute uses_duration;
    @attribute uses_events;
    @attribute duration;
    @attribute npvControls;
    @attribute npvMonitors;
    @attribute npvLabels;
  }
  @dataset pvControls;
  @dataset pvMonitors;
  @dataset pvLabels;
}


//------------------ ConfigV3 ------------------
@h5schema ConfigV3
  [[version(0)]]
{
  @dataset config {
    @attribute events;
    @attribute uses_l3t_events;
    @attribute uses_duration;
    @attribute uses_events;
    @attribute duration;
    @attribute npvControls;
    @attribute npvMonitors;
  }
  @dataset pvControls;
  @dataset pvMonitors;
  @dataset pvLabels;
}


//------------------ ConfigV3 ------------------
@h5schema ConfigV3
  [[version(1)]]
{
  @dataset config {
    @attribute events;
    @attribute uses_l3t_events;
    @attribute uses_duration;
    @attribute uses_events;
    @attribute duration;
    @attribute npvControls;
    @attribute npvMonitors;
    @attribute npvLabels;
  }
  @dataset pvControls;
  @dataset pvMonitors;
  @dataset pvLabels;
}
} //- @package ControlData
