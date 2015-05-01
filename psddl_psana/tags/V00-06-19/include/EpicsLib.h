#ifndef PSDDL_PSANA_EPICSLIB_H
#define PSDDL_PSANA_EPICSLIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsLib.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <stdexcept>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/epics.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace Psana {
namespace EpicsLib {

/**
 *  Set of helper classes simplifying access to EPICS data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

/**
 *  Non-specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  numeric EPICS types and numeric resulting types.
 */
template <typename PVClass, typename ValueType>
struct EpicsValue {
  static ValueType value(const Epics::EpicsPvHeader& pv, int index) {
    return boost::numeric_cast<ValueType>(static_cast<const PVClass&>(pv).value(index));
  }
};

/**
 *  Specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  numeric EPICS types and std::string as result type.
 */
template <typename PVClass>
struct EpicsValue<PVClass, std::string> {
  static std::string value(const Epics::EpicsPvHeader& pv, int index) {
    return boost::lexical_cast<std::string>(static_cast<const PVClass&>(pv).value(index));
  }
};

/**
 *  Specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  string EPICS types and numeric result type.
 */
template <typename ValueType>
struct EpicsValue<Epics::EpicsPvCtrlString, ValueType> {
  static ValueType value(const Epics::EpicsPvHeader& pv, int index) {
    return boost::lexical_cast<ValueType>(
        static_cast<const Epics::EpicsPvCtrlString&>(pv).value(index));
  }
};

/**
 *  Specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  string EPICS types and numeric result type.
 */
template <typename ValueType>
struct EpicsValue<Epics::EpicsPvTimeString, ValueType> {
  static ValueType value(const Epics::EpicsPvHeader& pv, int index) {
    return boost::lexical_cast<ValueType>(
        static_cast<const Epics::EpicsPvTimeString&>(pv).value(index));
  }
};

/**
 *  Specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  string EPICS types and string result type.
 */
template <>
struct EpicsValue<Epics::EpicsPvCtrlString, std::string> {
  static std::string value(const Epics::EpicsPvHeader& pv, int index) {
    return static_cast<const Epics::EpicsPvCtrlString&>(pv).value(index);
  }
};

/**
 *  Specialized version of the class which extracts value from
 *  EPICS PV and converts it to requested type. This version works for
 *  string EPICS types and string result type.
 */
template <>
struct EpicsValue<Epics::EpicsPvTimeString, std::string> {
  static std::string value(const Epics::EpicsPvHeader& pv, int index) {
    return static_cast<const Epics::EpicsPvTimeString&>(pv).value(index);
  }
};


/**
 *  Function which extracts value from EPICS PV and converts it to requested type.
 *  
 *  @throw boost::numeric::bad_numeric_cast  in case of overflows 
 *  @throw boost::bad_lexical_cast   if string vale cannot be converted to number
 *  @throw std::invalid_argument   if PV has unexpected type
 */
template <typename ValueType>
ValueType getEpicsValue(const Epics::EpicsPvHeader& pv, int index)
{
  switch(pv.dbrType()) {
  case Epics::DBR_TIME_STRING:
    return EpicsValue<Epics::EpicsPvTimeString,ValueType>::value(pv, index);
  case Epics::DBR_TIME_SHORT:
    return EpicsValue<Epics::EpicsPvTimeShort,ValueType>::value(pv, index);
  case Epics::DBR_TIME_FLOAT:
    return EpicsValue<Epics::EpicsPvTimeFloat,ValueType>::value(pv, index);
  case Epics::DBR_TIME_ENUM:
    return EpicsValue<Epics::EpicsPvTimeEnum,ValueType>::value(pv, index);
  case Epics::DBR_TIME_CHAR:
    return EpicsValue<Epics::EpicsPvTimeChar,ValueType>::value(pv, index);
  case Epics::DBR_TIME_LONG:
    return EpicsValue<Epics::EpicsPvTimeLong,ValueType>::value(pv, index);
  case Epics::DBR_TIME_DOUBLE:
    return EpicsValue<Epics::EpicsPvTimeDouble,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_STRING:
    return EpicsValue<Epics::EpicsPvCtrlString,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_SHORT:
    return EpicsValue<Epics::EpicsPvCtrlShort,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_FLOAT:
    return EpicsValue<Epics::EpicsPvCtrlFloat,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_ENUM:
    return EpicsValue<Epics::EpicsPvCtrlEnum,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_CHAR:
    return EpicsValue<Epics::EpicsPvCtrlChar,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_LONG:
    return EpicsValue<Epics::EpicsPvCtrlLong,ValueType>::value(pv, index);
  case Epics::DBR_CTRL_DOUBLE:
    return EpicsValue<Epics::EpicsPvCtrlDouble,ValueType>::value(pv, index);
  default:
    throw std::invalid_argument("EpicsLib::getEpicsValue - unexpected PV type: " +
        boost::lexical_cast<std::string>(pv.dbrType()));
  }
}


/**
 * Type traits for epics DBR types. DBR can be one of the dbr_time_xxx or dbr_ctrl_xxx types.
 */
template <typename DBR>
struct DBRTypeTraits {  
};

template <>
struct DBRTypeTraits<Epics::dbr_time_string> {
  typedef char value_type[Epics::MAX_STRING_SIZE];
  typedef Epics::EpicsPvTimeString pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_enum> {
  typedef uint16_t value_type;
  typedef Epics::EpicsPvTimeEnum pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_char> {
  typedef uint8_t value_type;
  typedef Epics::EpicsPvTimeChar pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_short> {
  typedef int16_t value_type;
  typedef Epics::EpicsPvTimeShort pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_long> {
  typedef int32_t value_type;
  typedef Epics::EpicsPvTimeLong pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_float> {
  typedef float value_type;
  typedef Epics::EpicsPvTimeFloat pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_time_double> {
  typedef double value_type;
  typedef Epics::EpicsPvTimeDouble pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_sts_string> {
  typedef char value_type[Epics::MAX_STRING_SIZE];
  typedef Epics::EpicsPvCtrlString pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_enum> {
  typedef uint16_t value_type;
  typedef Epics::EpicsPvCtrlEnum pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_char> {
  typedef uint8_t value_type;
  typedef Epics::EpicsPvCtrlChar pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_short> {
  typedef int16_t value_type;
  typedef Epics::EpicsPvCtrlShort pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_long> {
  typedef int32_t value_type;
  typedef Epics::EpicsPvCtrlLong pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_float> {
  typedef float value_type;
  typedef Epics::EpicsPvCtrlFloat pv_type;
};

template <>
struct DBRTypeTraits<Epics::dbr_ctrl_double> {
  typedef double value_type;
  typedef Epics::EpicsPvCtrlDouble pv_type;
};


/**
 * Type traits for epics PV types. PV can be one of the Epics::EpicsPvXxx types.
 */
template <typename PV>
struct PVTypeTraits {  
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeString> {
  typedef char value_type[Epics::MAX_STRING_SIZE];
  typedef Epics::dbr_time_string dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeEnum> {
  typedef uint16_t value_type;
  typedef Epics::dbr_time_enum dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeChar> {
  typedef uint8_t value_type;
  typedef Epics::dbr_time_char dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeShort> {
  typedef int16_t value_type;
  typedef Epics::dbr_time_short dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeLong> {
  typedef int32_t value_type;
  typedef Epics::dbr_time_long dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeFloat> {
  typedef float value_type;
  typedef Epics::dbr_time_float dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvTimeDouble> {
  typedef double value_type;
  typedef Epics::dbr_time_double dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlString> {
  typedef char value_type[Epics::MAX_STRING_SIZE];
  typedef Epics::dbr_sts_string dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlEnum> {
  typedef uint16_t value_type;
  typedef Epics::dbr_ctrl_enum dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlChar> {
  typedef uint8_t value_type;
  typedef Epics::dbr_ctrl_char dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlShort> {
  typedef int16_t value_type;
  typedef Epics::dbr_ctrl_short dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlLong> {
  typedef int32_t value_type;
  typedef Epics::dbr_ctrl_long dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlFloat> {
  typedef float value_type;
  typedef Epics::dbr_ctrl_float dbr_type;
};

template <>
struct PVTypeTraits<Epics::EpicsPvCtrlDouble> {
  typedef double value_type;
  typedef Epics::dbr_ctrl_double dbr_type;
};


} // namespace EpicsLib
} // namespace Psana

#endif // PSDDL_PSANA_EPICSLIB_H
