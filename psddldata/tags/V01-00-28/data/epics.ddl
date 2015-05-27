@package Epics  {
  @const int32_t iXtcVersion = 1;
  /* Maximum size of PV name string. */
  @const int32_t iMaxPvNameLength = 64;
  /* Maximum length of strings in EPICS. */
  @const int32_t MAX_STRING_SIZE = 40;
  /* Maximum lenght of units strings. */
  @const int32_t MAX_UNITS_SIZE = 8;
  /* Maximum length of strings specifying ENUMs. */
  @const int32_t MAX_ENUM_STRING_SIZE = 26;
  /* Maximum number of different ENUM constants. */
  @const int32_t MAX_ENUM_STATES = 16;
  /* Enum specifying type of DBR structures. */
  @enum DbrTypes (int32_t) {
    DBR_STRING = 0,
    DBR_SHORT = 1,
    DBR_FLOAT = 2,
    DBR_ENUM = 3,
    DBR_CHAR = 4,
    DBR_LONG = 5,
    DBR_DOUBLE = 6,
    DBR_STS_STRING = 7,
    DBR_STS_SHORT = 8,
    DBR_STS_FLOAT = 9,
    DBR_STS_ENUM = 10,
    DBR_STS_CHAR = 11,
    DBR_STS_LONG = 12,
    DBR_STS_DOUBLE = 13,
    DBR_TIME_STRING = 14,
    DBR_TIME_INT = 15,
    DBR_TIME_SHORT = 15,
    DBR_TIME_FLOAT = 16,
    DBR_TIME_ENUM = 17,
    DBR_TIME_CHAR = 18,
    DBR_TIME_LONG = 19,
    DBR_TIME_DOUBLE = 20,
    DBR_GR_STRING = 21,
    DBR_GR_SHORT = 22,
    DBR_GR_FLOAT = 23,
    DBR_GR_ENUM = 24,
    DBR_GR_CHAR = 25,
    DBR_GR_LONG = 26,
    DBR_GR_DOUBLE = 27,
    DBR_CTRL_STRING = 28,
    DBR_CTRL_SHORT = 29,
    DBR_CTRL_FLOAT = 30,
    DBR_CTRL_ENUM = 31,
    DBR_CTRL_CHAR = 32,
    DBR_CTRL_LONG = 33,
    DBR_CTRL_DOUBLE = 34,
  }


//------------------ epicsTimeStamp ------------------
/* EPICS timestamp type, includes seconds and nanoseconds.
           EPICS epoch corresponds to 1990-01-01 00:00:00Z. */
@type epicsTimeStamp
  [[value_type]]
{
  uint32_t _secPastEpoch -> sec;	/* Seconds since Jan 1, 1990 00:00 UTC */
  uint32_t _nsec -> nsec;	/* Nanoseconds within second. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ dbr_time_string ------------------
@type dbr_time_string
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_STRING;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)  [[inline]];

}


//------------------ dbr_time_short ------------------
@type dbr_time_short
  [[value_type]]
  [[pack(2)]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_SHORT;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;
  int16_t RISC_pad;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)
    RISC_pad(0)  [[inline]];

}


//------------------ dbr_time_float ------------------
@type dbr_time_float
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_FLOAT;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)  [[inline]];

}


//------------------ dbr_time_enum ------------------
@type dbr_time_enum
  [[value_type]]
  [[pack(2)]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_ENUM;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;
  int16_t RISC_pad;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)
    RISC_pad(0)  [[inline]];

}


//------------------ dbr_time_char ------------------
@type dbr_time_char
  [[value_type]]
  [[pack(1)]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_CHAR;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;
  int16_t RISC_pad0;
  uint8_t RISC_pad1;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)
    RISC_pad1(0), RISC_pad1(0)  [[inline]];

}


//------------------ dbr_time_long ------------------
@type dbr_time_long
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_LONG;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)  [[inline]];

}


//------------------ dbr_time_double ------------------
@type dbr_time_double
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_TIME_DOUBLE;

  int16_t _status -> status;
  int16_t _severity -> severity;
  epicsTimeStamp _stamp -> stamp;
  int32_t RISC_pad;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, stamp -> _stamp)
    RISC_pad(0)  [[inline]];

}


//------------------ dbr_sts_string ------------------
@type dbr_sts_string
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_STRING;

  int16_t _status -> status;
  int16_t _severity -> severity;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity)  [[inline]];

}


//------------------ dbr_ctrl_short ------------------
@type dbr_ctrl_short
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_SHORT;

  int16_t _status -> status;
  int16_t _severity -> severity;
  char _units[MAX_UNITS_SIZE] -> units;
  int16_t _upper_disp_limit -> upper_disp_limit;
  int16_t _lower_disp_limit -> lower_disp_limit;
  int16_t _upper_alarm_limit -> upper_alarm_limit;
  int16_t _upper_warning_limit -> upper_warning_limit;
  int16_t _lower_warning_limit -> lower_warning_limit;
  int16_t _lower_alarm_limit -> lower_alarm_limit;
  int16_t _upper_ctrl_limit -> upper_ctrl_limit;
  int16_t _lower_ctrl_limit -> lower_ctrl_limit;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, units -> _units, upper_disp_limit -> _upper_disp_limit, 
      lower_disp_limit -> _lower_disp_limit, upper_alarm_limit -> _upper_alarm_limit, 
      upper_warning_limit -> _upper_warning_limit, lower_warning_limit -> _lower_warning_limit, 
      lower_alarm_limit -> _lower_alarm_limit, upper_ctrl_limit -> _upper_ctrl_limit, 
      lower_ctrl_limit -> _lower_ctrl_limit)  [[inline]];

}


//------------------ dbr_ctrl_float ------------------
@type dbr_ctrl_float
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_FLOAT;

  int16_t _status -> status;
  int16_t _severity -> severity;
  int16_t _precision -> precision;
  int16_t RISC_pad;
  char _units[MAX_UNITS_SIZE] -> units;
  float _upper_disp_limit -> upper_disp_limit;
  float _lower_disp_limit -> lower_disp_limit;
  float _upper_alarm_limit -> upper_alarm_limit;
  float _upper_warning_limit -> upper_warning_limit;
  float _lower_warning_limit -> lower_warning_limit;
  float _lower_alarm_limit -> lower_alarm_limit;
  float _upper_ctrl_limit -> upper_ctrl_limit;
  float _lower_ctrl_limit -> lower_ctrl_limit;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, precision -> _precision, units -> _units, 
      upper_disp_limit -> _upper_disp_limit, lower_disp_limit -> _lower_disp_limit, 
      upper_alarm_limit -> _upper_alarm_limit, upper_warning_limit -> _upper_warning_limit, 
      lower_warning_limit -> _lower_warning_limit, lower_alarm_limit -> _lower_alarm_limit, 
      upper_ctrl_limit -> _upper_ctrl_limit, lower_ctrl_limit -> _lower_ctrl_limit)
    RISC_pad(0)  [[inline]];

}


//------------------ dbr_ctrl_enum ------------------
@type dbr_ctrl_enum
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_ENUM;

  int16_t _status -> status;
  int16_t _severity -> severity;
  int16_t _no_str -> no_str;
  char _strs[MAX_ENUM_STATES][MAX_ENUM_STRING_SIZE] -> strings;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, no_str -> _no_str, strings -> _strs)  [[inline]];

}


//------------------ dbr_ctrl_char ------------------
@type dbr_ctrl_char
  [[value_type]]
  [[pack(1)]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_CHAR;

  int16_t _status -> status;
  int16_t _severity -> severity;
  char _units[MAX_UNITS_SIZE] -> units;
  uint8_t _upper_disp_limit -> upper_disp_limit;
  uint8_t _lower_disp_limit -> lower_disp_limit;
  uint8_t _upper_alarm_limit -> upper_alarm_limit;
  uint8_t _upper_warning_limit -> upper_warning_limit;
  uint8_t _lower_warning_limit -> lower_warning_limit;
  uint8_t _lower_alarm_limit -> lower_alarm_limit;
  uint8_t _upper_ctrl_limit -> upper_ctrl_limit;
  uint8_t _lower_ctrl_limit -> lower_ctrl_limit;
  uint8_t RISC_pad;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, units -> _units, 
      upper_disp_limit -> _upper_disp_limit, lower_disp_limit -> _lower_disp_limit, 
      upper_alarm_limit -> _upper_alarm_limit, upper_warning_limit -> _upper_warning_limit, 
      lower_warning_limit -> _lower_warning_limit, lower_alarm_limit -> _lower_alarm_limit, 
      upper_ctrl_limit -> _upper_ctrl_limit, lower_ctrl_limit -> _lower_ctrl_limit)
    RISC_pad(0)  [[inline]];

}


//------------------ dbr_ctrl_long ------------------
@type dbr_ctrl_long
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_LONG;

  int16_t _status -> status;
  int16_t _severity -> severity;
  char _units[MAX_UNITS_SIZE] -> units;
  int32_t _upper_disp_limit -> upper_disp_limit;
  int32_t _lower_disp_limit -> lower_disp_limit;
  int32_t _upper_alarm_limit -> upper_alarm_limit;
  int32_t _upper_warning_limit -> upper_warning_limit;
  int32_t _lower_warning_limit -> lower_warning_limit;
  int32_t _lower_alarm_limit -> lower_alarm_limit;
  int32_t _upper_ctrl_limit -> upper_ctrl_limit;
  int32_t _lower_ctrl_limit -> lower_ctrl_limit;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, units -> _units, 
      upper_disp_limit -> _upper_disp_limit, lower_disp_limit -> _lower_disp_limit, 
      upper_alarm_limit -> _upper_alarm_limit, upper_warning_limit -> _upper_warning_limit, 
      lower_warning_limit -> _lower_warning_limit, lower_alarm_limit -> _lower_alarm_limit, 
      upper_ctrl_limit -> _upper_ctrl_limit, lower_ctrl_limit -> _lower_ctrl_limit)  [[inline]];

}


//------------------ dbr_ctrl_double ------------------
@type dbr_ctrl_double
  [[value_type]]
{
  @const int32_t DBR_TYPE_ID = DBR_CTRL_DOUBLE;

  int16_t _status -> status;
  int16_t _severity -> severity;
  int16_t _precision -> precision;
  int16_t RISC_pad0;
  char _units[MAX_UNITS_SIZE] -> units;
  double _upper_disp_limit -> upper_disp_limit;
  double _lower_disp_limit -> lower_disp_limit;
  double _upper_alarm_limit -> upper_alarm_limit;
  double _upper_warning_limit -> upper_warning_limit;
  double _lower_warning_limit -> lower_warning_limit;
  double _lower_alarm_limit -> lower_alarm_limit;
  double _upper_ctrl_limit -> upper_ctrl_limit;
  double _lower_ctrl_limit -> lower_ctrl_limit;

  /* Constructor which takes values for every attribute */
  @init(status -> _status, severity -> _severity, precision -> _precision, units -> _units, 
      upper_disp_limit -> _upper_disp_limit, lower_disp_limit -> _lower_disp_limit, 
      upper_alarm_limit -> _upper_alarm_limit, upper_warning_limit -> _upper_warning_limit, 
      lower_warning_limit -> _lower_warning_limit, lower_alarm_limit -> _lower_alarm_limit, 
      upper_ctrl_limit -> _upper_ctrl_limit, lower_ctrl_limit -> _lower_ctrl_limit)
    RISC_pad0(0)  [[inline]];

}


//------------------ EpicsPvHeader ------------------
/* Base class for EPICS data types stored in XTC files. */
@type EpicsPvHeader
{
  int16_t _iPvId -> pvId;	/* PV ID number assigned by DAQ. */
  int16_t _iDbrType -> dbrType;	/* DBR structure type. */
  int16_t _iNumElements -> numElements;	/* Number of elements in EPICS DBR structure */

  /* Returns 1 if PV is one of CTRL types, 0 otherwise. */
  uint8_t isCtrl()
  [[language("C++")]] @{ return @self.dbrType() >= DBR_CTRL_STRING and @self.dbrType() <= DBR_CTRL_DOUBLE; @}

  /* Returns 1 if PV is one of TIME types, 0 otherwise. */
  uint8_t isTime()
  [[language("C++")]] @{ return @self.dbrType() >= DBR_TIME_STRING and @self.dbrType() <= DBR_TIME_DOUBLE; @}

  /* Returns status value for the PV. */
  uint16_t status() [[external]];

  /* Returns severity value for the PV. */
  uint16_t severity() [[external]];

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlHeader ------------------
/* Base class for all CTRL-type PV values. */
@type EpicsPvCtrlHeader(EpicsPvHeader)
{
  char _sPvName[iMaxPvNameLength] -> pvName  [[shape_method(None)]];	/* PV name. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeHeader ------------------
/* Base class for all TIME-type PV values. */
@type EpicsPvTimeHeader(EpicsPvHeader)
{

  /* EPICS timestamp value. */
  epicsTimeStamp stamp() [[external]];

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlString ------------------
@type EpicsPvCtrlString(EpicsPvCtrlHeader)
{
  dbr_sts_string _dbr -> dbr;
  char _data[@self.numElements()][ MAX_STRING_SIZE] -> data;

  string value(uint32_t i)
  [[language("C++")]] @{ return data(i); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlShort ------------------
@type EpicsPvCtrlShort(EpicsPvCtrlHeader)
{
  dbr_ctrl_short _dbr -> dbr;
  int16_t _data[@self.numElements()] -> data;

  int16_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlFloat ------------------
@type EpicsPvCtrlFloat(EpicsPvCtrlHeader)
{
  int16_t pad0;
  dbr_ctrl_float _dbr -> dbr;
  float _data[@self.numElements()] -> data;

  float value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlEnum ------------------
@type EpicsPvCtrlEnum(EpicsPvCtrlHeader)
{
  dbr_ctrl_enum _dbr -> dbr;
  uint16_t _data[@self.numElements()] -> data;

  uint16_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlChar ------------------
@type EpicsPvCtrlChar(EpicsPvCtrlHeader)
{
  dbr_ctrl_char _dbr -> dbr;
  uint8_t _data[@self.numElements()] -> data;

  uint8_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlLong ------------------
@type EpicsPvCtrlLong(EpicsPvCtrlHeader)
{
  int16_t pad0;
  dbr_ctrl_long _dbr -> dbr;
  int32_t _data[@self.numElements()] -> data;

  int32_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvCtrlDouble ------------------
@type EpicsPvCtrlDouble(EpicsPvCtrlHeader)
{
  int16_t pad0;
  dbr_ctrl_double _dbr -> dbr;
  double _data[@self.numElements()] -> data;

  double value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeString ------------------
@type EpicsPvTimeString(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_string _dbr -> dbr;
  char _data[@self.numElements()][ MAX_STRING_SIZE] -> data;

  string value(uint32_t i)
  [[language("C++")]] @{ return data(i); @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeShort ------------------
@type EpicsPvTimeShort(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_short _dbr -> dbr;
  int16_t _data[@self.numElements()] -> data;

  int16_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeFloat ------------------
@type EpicsPvTimeFloat(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_float _dbr -> dbr;
  float _data[@self.numElements()] -> data;

  float value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeEnum ------------------
@type EpicsPvTimeEnum(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_enum _dbr -> dbr;
  uint16_t _data[@self.numElements()] -> data;

  uint16_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeChar ------------------
@type EpicsPvTimeChar(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_char _dbr -> dbr;
  uint8_t _data[@self.numElements()] -> data;

  uint8_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeLong ------------------
@type EpicsPvTimeLong(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_long _dbr -> dbr;
  int32_t _data[@self.numElements()] -> data;

  int32_t value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ EpicsPvTimeDouble ------------------
@type EpicsPvTimeDouble(EpicsPvTimeHeader)
{
  int16_t pad0;
  dbr_time_double _dbr -> dbr;
  double _data[@self.numElements()] -> data;

  double value(uint32_t i)
  [[language("C++")]] @{ return data()[i]; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ PvConfigV1 ------------------
@type PvConfigV1
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t iMaxPvDescLength = 64;

  int16_t iPvId -> pvId;
  char sPvDesc[iMaxPvDescLength] -> description  [[shape_method(None)]];
  int16_t _pad0;
  float fInterval -> interval;

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_EpicsConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  int32_t _iNumPv -> numPv;
  PvConfigV1 _pvConfig[@self.numPv()] -> getPvConfig;

  /* Constructor which just takes the number of Pvs contained */
  @init(numPv -> _iNumPv)  [[inline]];

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package Epics
