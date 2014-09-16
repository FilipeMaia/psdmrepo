@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/ClockTime.hh")]];
@package ControlData  {


//------------------ PVControl ------------------
@type PVControl
  [[value_type]]
  [[config_type]]
  [[pack(4)]]
{
  /* Length of the name array. */
  @const int32_t NameSize = 32;
  /* Special value used for _index when PV is not an array */
  @const int32_t NoArray = 0xFFFFFFFF;

  char _name[32] -> name  [[shape_method(None)]];	/* Name of the control. */
  uint32_t _index -> index;	/* Index of the control PV (for arrays) or NoArray. */
  double _value -> value;	/* Value for this control. */

  /* Returns true if the control is an array. */
  uint8_t array()
  [[language("C++")]] @{ return _index != NoArray; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ PVMonitor ------------------
@type PVMonitor
  [[value_type]]
  [[pack(4)]]
{
  /* Length of the name array. */
  @const int32_t NameSize = 32;
  /* Special value used for _index when PV is not an array */
  @const int32_t NoArray = 0xFFFFFFFF;

  char _name[32] -> name  [[shape_method(None)]];	/* Name of the control. */
  uint32_t _index -> index;	/* Index of the control PV (for arrays) or NoArray. */
  double _loValue -> loValue;	/* Lowest value for this monitor. */
  double _hiValue -> hiValue;	/* Highest value for this monitor. */

  /* Returns true if the monitor is an array. */
  uint8_t array()
  [[language("C++")]] @{ return _index != NoArray; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ PVLabel ------------------
@type PVLabel
  [[value_type]]
  [[pack(4)]]
{
  /* Length of the PV name array. */
  @const int32_t NameSize = 32;
  /* Length of the value array. */
  @const int32_t ValueSize = 64;

  char _name[32] -> name  [[shape_method(None)]];	/* PV name. */
  char _value[64] -> value  [[shape_method(None)]];	/* Label value. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_ControlConfig, 1)]]
  [[config_type]]
{
  uint32_t _control {
    uint32_t _bf_events:30 -> events;	/* Maximum number of events per scan. */
    uint8_t _bf_uses_duration:1 -> uses_duration;	/* returns true if the configuration uses duration control. */
    uint8_t _bf_uses_events:1 -> uses_events;	/* returns true if the configuration uses events limit. */
  }
  uint32_t _reserved;
  Pds.ClockTime _duration -> duration;	/* Maximum duration of the scan. */
  uint32_t _npvControls -> npvControls;	/* Number of PVControl objects in this configuration. */
  uint32_t _npvMonitors -> npvMonitors;	/* Number of PVMonitor objects in this configuration. */
  PVControl _pvControls[@self.npvControls()] -> pvControls  [[shape_method(pvControls_shape)]];	/* PVControl configuration objects */
  PVMonitor _pvMonitors[@self.npvMonitors()] -> pvMonitors  [[shape_method(pvMonitors_shape)]];	/* PVMonitor configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_ControlConfig, 2)]]
  [[config_type]]
{
  uint32_t _control {
    uint32_t _bf_events:30 -> events;	/* Maximum number of events per scan. */
    uint8_t _bf_uses_duration:1 -> uses_duration;	/* returns true if the configuration uses duration control. */
    uint8_t _bf_uses_events:1 -> uses_events;	/* returns true if the configuration uses events limit. */
  }
  uint32_t _reserved;
  Pds.ClockTime _duration -> duration;	/* Maximum duration of the scan. */
  uint32_t _npvControls -> npvControls;	/* Number of PVControl objects in this configuration. */
  uint32_t _npvMonitors -> npvMonitors;	/* Number of PVMonitor objects in this configuration. */
  uint32_t _npvLabels -> npvLabels;	/* Number of PVLabel objects in this configuration. */
  PVControl _pvControls[@self.npvControls()] -> pvControls  [[shape_method(pvControls_shape)]];	/* PVControl configuration objects */
  PVMonitor _pvMonitors[@self.npvMonitors()] -> pvMonitors  [[shape_method(pvMonitors_shape)]];	/* PVMonitor configuration objects */
  PVLabel _pvLabels[@self.npvLabels()] -> pvLabels  [[shape_method(pvLabels_shape)]];	/* PVLabel configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ ConfigV3 ------------------
@type ConfigV3
  [[type_id(Id_ControlConfig, 3)]]
  [[config_type]]
{
  uint32_t _control {
    uint32_t _bf_events:29 -> events;	/* Maximum number of events per scan. */
    uint8_t _bf_uses_l3t_events:1 -> uses_l3t_events;	/* returns true if the configuration uses l3trigger events limit. */
    uint8_t _bf_uses_duration:1 -> uses_duration;	/* returns true if the configuration uses duration control. */
    uint8_t _bf_uses_events:1 -> uses_events;	/* returns true if the configuration uses events limit. */
  }
  uint32_t _reserved;
  Pds.ClockTime _duration -> duration;	/* Maximum duration of the scan. */
  uint32_t _npvControls -> npvControls;	/* Number of PVControl objects in this configuration. */
  uint32_t _npvMonitors -> npvMonitors;	/* Number of PVMonitor objects in this configuration. */
  uint32_t _npvLabels -> npvLabels;	/* Number of PVLabel objects in this configuration. */
  PVControl _pvControls[@self.npvControls()] -> pvControls  [[shape_method(pvControls_shape)]];	/* PVControl configuration objects */
  PVMonitor _pvMonitors[@self.npvMonitors()] -> pvMonitors  [[shape_method(pvMonitors_shape)]];	/* PVMonitor configuration objects */
  PVLabel _pvLabels[@self.npvLabels()] -> pvLabels  [[shape_method(pvLabels_shape)]];	/* PVLabel configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package ControlData
