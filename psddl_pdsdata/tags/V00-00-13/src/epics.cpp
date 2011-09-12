//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class epics...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pdsdata/epics.ddl.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <stdio.h>
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PsddlPds;

namespace {
    
  /*
   * Imported from Epics header file: ${EPICS_PROJECT_DIR}/base/current/include/alarm.h
   *  
   */
  const char *epicsAlarmSeverityStrings [] = 
  {
      "NO_ALARM",
      "MINOR",
      "MAJOR",
      "INVALID",
  };
  
  const char *epicsAlarmConditionStrings [] = 
  {
      "NO_ALARM",
      "READ",
      "WRITE",
      "HIHI",
      "HIGH",
      "LOLO",
      "LOW",
      "STATE",
      "COS",
      "COMM",
      "TIMEOUT",
      "HWLIMIT",
      "CALC",
      "SCAN",
      "LINK",
      "SOFT",
      "BAD_SUB",
      "UDF",
      "DISABLE",
      "SIMM",
      "READ_ACCESS",
      "WRITE_ACCESS",
  };    

  /*
   * Imported from Epics source file: ${EPICS_PROJECT_DIR}/base/current/src/ca/access.cpp
   *  
   */
  const char *dbr_text[] = 
  {
      "DBR_STRING",
      "DBR_SHORT",
      "DBR_FLOAT",
      "DBR_ENUM",
      "DBR_CHAR",
      "DBR_LONG",
      "DBR_DOUBLE",
      "DBR_STS_STRING",
      "DBR_STS_SHORT",
      "DBR_STS_FLOAT",
      "DBR_STS_ENUM",
      "DBR_STS_CHAR",
      "DBR_STS_LONG",
      "DBR_STS_DOUBLE",
      "DBR_TIME_STRING",
      "DBR_TIME_SHORT",
      "DBR_TIME_FLOAT",
      "DBR_TIME_ENUM",
      "DBR_TIME_CHAR",
      "DBR_TIME_LONG",
      "DBR_TIME_DOUBLE",
      "DBR_GR_STRING",
      "DBR_GR_SHORT",
      "DBR_GR_FLOAT",
      "DBR_GR_ENUM",
      "DBR_GR_CHAR",
      "DBR_GR_LONG",
      "DBR_GR_DOUBLE",
      "DBR_CTRL_STRING",
      "DBR_CTRL_SHORT",
      "DBR_CTRL_FLOAT",
      "DBR_CTRL_ENUM",
      "DBR_CTRL_CHAR",
      "DBR_CTRL_LONG",
      "DBR_CTRL_DOUBLE",
  };
  
  void printValue( const char* value ) { printf( "%s", value ); }
  void printValue( int16_t value ) { printf( "%d", int(value) ); }
  void printValue( int32_t value ) { printf( "%ld", long(value) ); }
  void printValue( float value ) { printf( "%f", value ); }
  void printValue( double value ) { printf( "%lf", value ); }
  void printValue( uint16_t value ) { printf( "%d", int(value) ); }
  void printValue( uint8_t value ) { printf( "%d", int(value) ); }

  template <typename DBR>
  void printTimeDbr(const DBR& epics)
  {
    printf( "\n> PV (Time) Id %d\n", epics.pvId()); 
    printf( "Type: %s\n", ::dbr_text[epics.dbrType()] ); 
    if ( epics.numElements() > 1 ) printf("Length: %d\n", epics.numElements());
    
    printf("Status: %s\n", ::epicsAlarmConditionStrings[epics.dbr().status()] );
    printf("Severity: %s\n", ::epicsAlarmSeverityStrings[epics.dbr().severity()] );    
    
    static const char timeFormatStr[40] = "%Y-%m-%d %H:%M:%S"; /* Time format string */    
    char sTimeText[40];
    
    // Epics Epoch starts from 1990, whereas linux time.h Epoch starts from 1970
    time_t secs = epics.dbr().stamp().sec() + (20*365+5)*24*3600;
    struct tm tmTimeStamp;
    localtime_r( &secs, &tmTimeStamp );
    
    strftime(sTimeText, sizeof(sTimeText), timeFormatStr, &tmTimeStamp );
  
    printf( "TimeStamp: %s.%09u\n", sTimeText, (unsigned int) epics.dbr().stamp().nsec() );
   
    printf( "Value: " );
    for ( int iElement = 0; iElement < epics.numElements(); iElement++ )
    {
        printValue(epics.value(iElement));
        if ( iElement < epics.numElements()-1 ) printf( ", " );
    }
    printf( "\n" );     
  
  }

  template <class TCtrl> // Default not to print precision field
  void printPrecisionField(TCtrl& pvCtrlVal)  {}

  void printPrecisionField(const Epics::dbr_ctrl_double& pvCtrlVal)
  {
      printf( "Precision: %d\n", pvCtrlVal.precision() );
  }

  void printPrecisionField(const Epics::dbr_ctrl_float& pvCtrlVal)
  {
      printf( "Precision: %d\n", pvCtrlVal.precision() );    
  }


  template <class TCtrl> inline
  void printCtrlFields(TCtrl& pvCtrlVal)
  {
      printPrecisionField(pvCtrlVal);
      printf( "Units: %s\n", pvCtrlVal.units() );        
      
      printf( "Hi Disp : " );
      ::printValue(pvCtrlVal.upper_disp_limit());
      printf( "  Lo Disp : " );
      ::printValue(pvCtrlVal.lower_disp_limit());
      printf( "\n" );
      
      printf( "Hi Alarm: " );
      ::printValue(pvCtrlVal.upper_alarm_limit());
      printf( "  Hi Warn : " );
      ::printValue(pvCtrlVal.upper_warning_limit());
      printf( "\n" );
      
      printf( "Lo Warn : " );
      ::printValue(pvCtrlVal.lower_warning_limit());
      printf( "  Lo Alarm: " );
      ::printValue(pvCtrlVal.lower_alarm_limit());
      printf( "\n" );
      
      printf( "Hi Ctrl : " );
      ::printValue(pvCtrlVal.upper_ctrl_limit());
      printf( "  Lo Ctrl : " );
      ::printValue(pvCtrlVal.lower_ctrl_limit());
      printf( "\n" );
  }

  void printCtrlFields(const Epics::dbr_sts_string& pvCtrlVal) 
  {
  }

  void printCtrlFields(const Epics::dbr_ctrl_enum& pvCtrlVal)
  {
      if ( pvCtrlVal.no_str() > 0 )
      {
          printf( "EnumState Num: %d\n", pvCtrlVal.no_str() );
          
          const char* strings = pvCtrlVal.strings();
          std::vector<int> shape = pvCtrlVal.strings_shape();
          for (int iEnumState = 0; iEnumState < pvCtrlVal.no_str(); iEnumState++ ) {
              printf( "EnumState[%d]: %s\n", iEnumState, strings );
              strings += shape[1];
          }
      }
  }
  

  template <typename DBR>
  void printCtrlDbr(const DBR& epics)
  {
    printf( "\n> PV (Ctrl) Id %d\n", epics.pvId() ); 
    printf( "Name: %s\n", epics.pvName() );
    printf( "Type: %s\n", ::dbr_text[epics.dbrType()] ); 
    if ( epics.numElements() > 1 ) printf("Length: %d\n", epics.numElements());
    
    printf("Status: %s\n", ::epicsAlarmConditionStrings[epics.dbr().status()] );
    printf("Severity: %s\n", ::epicsAlarmSeverityStrings[epics.dbr().severity()] );    
        
    ::printCtrlFields( epics.dbr() );
    
    printf( "Value: " );
    for ( int iElement = 0; iElement < epics.numElements(); iElement++ )
    {
        printValue(epics.value(iElement));
        if ( iElement < epics.numElements()-1 ) printf( ", " );
    }
    printf( "\n" );     
  }
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace PsddlPds {
namespace Epics {


void
EpicsPvHeader::print() const
{
  switch (this->dbrType()) {
  case DBR_TIME_STRING:
    printTimeDbr(*(const EpicsPvTimeString*)this);
    break;
  case DBR_TIME_SHORT:
    printTimeDbr(*(const EpicsPvTimeShort*)this);
    break;
  case DBR_TIME_FLOAT:
    printTimeDbr(*(const EpicsPvTimeFloat*)this);
    break;
  case DBR_TIME_ENUM:
    printTimeDbr(*(const EpicsPvTimeEnum*)this);
    break;
  case DBR_TIME_CHAR:
    printTimeDbr(*(const EpicsPvTimeChar*)this);
    break;
  case DBR_TIME_LONG:
    printTimeDbr(*(const EpicsPvTimeLong*)this);
    break;
  case DBR_TIME_DOUBLE:
    printTimeDbr(*(const EpicsPvTimeDouble*)this);
    break;
  case DBR_CTRL_STRING:
    printCtrlDbr(*(const EpicsPvCtrlString*)this);
    break;
  case DBR_CTRL_SHORT:
    printCtrlDbr(*(const EpicsPvCtrlShort*)this);
    break;
  case DBR_CTRL_FLOAT:
    printCtrlDbr(*(const EpicsPvCtrlFloat*)this);
    break;
  case DBR_CTRL_ENUM:
    printCtrlDbr(*(const EpicsPvCtrlEnum*)this);
    break;
  case DBR_CTRL_CHAR:
    printCtrlDbr(*(const EpicsPvCtrlChar*)this);
    break;
  case DBR_CTRL_LONG:
    printCtrlDbr(*(const EpicsPvCtrlLong*)this);
    break;
  case DBR_CTRL_DOUBLE:
    printCtrlDbr(*(const EpicsPvCtrlDouble*)this);
    break;
  }
}

uint16_t
EpicsPvHeader::status() const
{
  switch (this->dbrType()) {
  case DBR_TIME_STRING:
    return ((const EpicsPvTimeString*)this)->dbr().status();
    break;
  case DBR_TIME_SHORT:
    return ((const EpicsPvTimeShort*)this)->dbr().status();
    break;
  case DBR_TIME_FLOAT:
    return ((const EpicsPvTimeFloat*)this)->dbr().status();
    break;
  case DBR_TIME_ENUM:
    return ((const EpicsPvTimeEnum*)this)->dbr().status();
    break;
  case DBR_TIME_CHAR:
    return ((const EpicsPvTimeChar*)this)->dbr().status();
    break;
  case DBR_TIME_LONG:
    return ((const EpicsPvTimeLong*)this)->dbr().status();
    break;
  case DBR_TIME_DOUBLE:
    return ((const EpicsPvTimeDouble*)this)->dbr().status();
    break;
  case DBR_CTRL_STRING:
    return ((const EpicsPvCtrlString*)this)->dbr().status();
    break;
  case DBR_CTRL_SHORT:
    return ((const EpicsPvCtrlShort*)this)->dbr().status();
    break;
  case DBR_CTRL_FLOAT:
    return ((const EpicsPvCtrlFloat*)this)->dbr().status();
    break;
  case DBR_CTRL_ENUM:
    return ((const EpicsPvCtrlEnum*)this)->dbr().status();
    break;
  case DBR_CTRL_CHAR:
    return ((const EpicsPvCtrlChar*)this)->dbr().status();
    break;
  case DBR_CTRL_LONG:
    return ((const EpicsPvCtrlLong*)this)->dbr().status();
    break;
  case DBR_CTRL_DOUBLE:
    return ((const EpicsPvCtrlDouble*)this)->dbr().status();
    break;
  }
  return 0;
}

uint16_t
EpicsPvHeader::severity() const
{
  switch (this->dbrType()) {
  case DBR_TIME_STRING:
    return ((const EpicsPvTimeString*)this)->dbr().severity();
    break;
  case DBR_TIME_SHORT:
    return ((const EpicsPvTimeShort*)this)->dbr().severity();
    break;
  case DBR_TIME_FLOAT:
    return ((const EpicsPvTimeFloat*)this)->dbr().severity();
    break;
  case DBR_TIME_ENUM:
    return ((const EpicsPvTimeEnum*)this)->dbr().severity();
    break;
  case DBR_TIME_CHAR:
    return ((const EpicsPvTimeChar*)this)->dbr().severity();
    break;
  case DBR_TIME_LONG:
    return ((const EpicsPvTimeLong*)this)->dbr().severity();
    break;
  case DBR_TIME_DOUBLE:
    return ((const EpicsPvTimeDouble*)this)->dbr().severity();
    break;
  case DBR_CTRL_STRING:
    return ((const EpicsPvCtrlString*)this)->dbr().severity();
    break;
  case DBR_CTRL_SHORT:
    return ((const EpicsPvCtrlShort*)this)->dbr().severity();
    break;
  case DBR_CTRL_FLOAT:
    return ((const EpicsPvCtrlFloat*)this)->dbr().severity();
    break;
  case DBR_CTRL_ENUM:
    return ((const EpicsPvCtrlEnum*)this)->dbr().severity();
    break;
  case DBR_CTRL_CHAR:
    return ((const EpicsPvCtrlChar*)this)->dbr().severity();
    break;
  case DBR_CTRL_LONG:
    return ((const EpicsPvCtrlLong*)this)->dbr().severity();
    break;
  case DBR_CTRL_DOUBLE:
    return ((const EpicsPvCtrlDouble*)this)->dbr().severity();
    break;
  }
  return 0;
}

epicsTimeStamp
EpicsPvTimeHeader::stamp() const
{
  switch (this->dbrType()) {
  case DBR_TIME_STRING:
    return ((const EpicsPvTimeString*)this)->dbr().stamp();
    break;
  case DBR_TIME_SHORT:
    return ((const EpicsPvTimeShort*)this)->dbr().stamp();
    break;
  case DBR_TIME_FLOAT:
    return ((const EpicsPvTimeFloat*)this)->dbr().stamp();
    break;
  case DBR_TIME_ENUM:
    return ((const EpicsPvTimeEnum*)this)->dbr().stamp();
    break;
  case DBR_TIME_CHAR:
    return ((const EpicsPvTimeChar*)this)->dbr().stamp();
    break;
  case DBR_TIME_LONG:
    return ((const EpicsPvTimeLong*)this)->dbr().stamp();
    break;
  case DBR_TIME_DOUBLE:
    return ((const EpicsPvTimeDouble*)this)->dbr().stamp();
    break;
  }
  return epicsTimeStamp();
}

} // namespace Epics
} // namespace PsddlPds
