#include "psana_test/epics2str.h"

#include <sstream>

namespace {

  template <class T>
  std::string obj2str( const T &obj) {
    return "* unknown type*";
  }

  template <>
  std::string obj2str<Pds::Epics::epicsTimeStamp>( const Pds::Epics::epicsTimeStamp &obj) {
    std::ostringstream str;
    str << "sec=" << obj.sec() << " nsec=" << obj.nsec();
    return str.str();
  }  

  template <class T>
  std::string dbrTime2str(const T &obj) {
    std::ostringstream str;
    str << "status=" << obj.status() << " severity=" << obj.severity();
    str << " " << obj2str(obj.stamp());
    return str.str();
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_string>(const Pds::Epics::dbr_time_string &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_string>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_short>(const Pds::Epics::dbr_time_short &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_short>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_float>(const Pds::Epics::dbr_time_float &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_float>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_enum>(const Pds::Epics::dbr_time_enum &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_enum>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_char>(const Pds::Epics::dbr_time_char &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_char>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_long>(const Pds::Epics::dbr_time_long &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_long>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_time_double>(const Pds::Epics::dbr_time_double &obj) {
    return dbrTime2str<Pds::Epics::dbr_time_double>(obj);
  }  

  template <class T>
  std::string dbrCtrlNumericWithoutPrecision2str(const T &obj) {
    std::ostringstream str;
    str.precision(4);
    str << "status=" << obj.status() << " severity=" << obj.severity();
    str << " units=" << obj.units();
    str << " upper_disp_limit=" << std::scientific << obj.upper_disp_limit();
    str << " lower_disp_limit=" << std::scientific << obj.lower_disp_limit();
    str << " upper_alarm_limit=" << std::scientific << obj.upper_alarm_limit();
    str << " upper_warning_limit=" << std::scientific << obj.upper_warning_limit();
    str << " lower_warning_limit=" << std::scientific << obj.lower_warning_limit();
    str << " lower_alarm_limit=" << std::scientific << obj.lower_alarm_limit();
    str << " upper_ctrl_limit=" << std::scientific << obj.upper_ctrl_limit();
    str << " lower_ctrl_limit=" << std::scientific << obj.lower_ctrl_limit();
    return str.str();
  }

  template <>
  std::string obj2str<Pds::Epics::dbr_sts_string>(const Pds::Epics::dbr_sts_string &obj) {
    std::ostringstream str;
    str << "status=" << obj.status() << " severity=" << obj.severity();
    return str.str();
  }
    
  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_short>(const Pds::Epics::dbr_ctrl_short &obj) {
    return dbrCtrlNumericWithoutPrecision2str<Pds::Epics::dbr_ctrl_short>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_float>(const Pds::Epics::dbr_ctrl_float &obj) {
    std::ostringstream str;
    str << dbrCtrlNumericWithoutPrecision2str<Pds::Epics::dbr_ctrl_float>(obj);
    str << " precision=" << obj.precision();
    return str.str();
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_double>(const Pds::Epics::dbr_ctrl_double &obj) {
    std::ostringstream str;
    str << dbrCtrlNumericWithoutPrecision2str<Pds::Epics::dbr_ctrl_double>(obj);
    str << " precision=" << obj.precision();
    return str.str();
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_enum>(const Pds::Epics::dbr_ctrl_enum &obj) {
    std::ostringstream str;
    str << "status=" << obj.status() << " severity=" << obj.severity() << " no_str=" << obj.no_str();
    for (int idx=0; idx < obj.no_str(); ++idx) {
      str << " str[" << idx << "]=" << obj.strings(idx);
    }
    return str.str();
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_char>(const Pds::Epics::dbr_ctrl_char &obj) {
    return dbrCtrlNumericWithoutPrecision2str<Pds::Epics::dbr_ctrl_char>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::dbr_ctrl_long>(const Pds::Epics::dbr_ctrl_long &obj) {
    return dbrCtrlNumericWithoutPrecision2str<Pds::Epics::dbr_ctrl_long>(obj);
  }  

  template <>
  std::string obj2str<Pds::Epics::EpicsPvHeader>(const Pds::Epics::EpicsPvHeader &obj) {
    std::ostringstream str;
    str << "pvId=" << obj.pvId();
    str << " dbrType=" << obj.dbrType();
    str << " numElem=" << obj.numElements();
    str << " isCtrl=" << int(obj.isCtrl());
    str << " isTime=" << int(obj.isTime());
    // don't print severity and status by calling EpicsPvHeader::severity(), status().
    // These functions assume the EpicsPvHeader object is a sub piece of a epics pv
    // with dbr in it. These functions switch on the dbr type to go to the dbr data.
    return str.str();
  }

  template <>
  std::string obj2str<Pds::Epics::EpicsPvCtrlHeader>(const Pds::Epics::EpicsPvCtrlHeader &obj) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(obj);
    str << " pvName=" << obj.pvName();
    return str.str();
  }

  // omit specialization for EpicsPvTimeHeader - it just provides function to stamp() in the
  // dbr data, we will get it from there.

  template <class T>
  std::string val2str( const T &v) {
    std::ostringstream str;
    str.precision(4);
    str << std::scientific << v;
    return str.str();
  }

  template <>
  std::string val2str( const unsigned char &obj) {
    std::ostringstream str;
    str << int(obj);
    return str.str();
  }

  template <>
  std::string val2str( const char &obj) {
    std::ostringstream str;
    str << int(obj);
    return str.str();
  }

  template <class T>
  std::string epicsPvValues2str(const T &obj) {
    std::ostringstream str;
    const int maxToPrint = 10;
    str << "value(0)=" << val2str(obj.value(0));
    int numToPrint = std::min(int(obj.numElements()), maxToPrint);
    for (int idx=1; idx < numToPrint; ++idx) {
      str << " value(" << idx << ")=" << val2str(obj.value(idx));
    }
    if (obj.numElements()>maxToPrint) {
      str << " ...";
    }
    return str.str();
  }

}

namespace psana_test {

std::string epics2Str(Pds::Xtc *xtc) {
  std::ostringstream str;
  if (xtc->contains.id() != Pds::TypeId::Id_Epics) return str.str();
  if (xtc->damage.value() != 0) return str.str();
  Pds::Epics::EpicsPvHeader *pvHeader = 
    static_cast<Pds::Epics::EpicsPvHeader *>(static_cast<void *>(xtc->payload()));
  switch (pvHeader->dbrType()) {
  case Pds::Epics::DBR_CTRL_STRING:
    {
      Pds::Epics::EpicsPvCtrlString *p = 
        static_cast<Pds::Epics::EpicsPvCtrlString *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlString2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_SHORT:
    {
      Pds::Epics::EpicsPvCtrlShort *p = 
        static_cast<Pds::Epics::EpicsPvCtrlShort *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlShort2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_FLOAT:
    {
      Pds::Epics::EpicsPvCtrlFloat *p = 
        static_cast<Pds::Epics::EpicsPvCtrlFloat *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlFloat2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_ENUM:
    {
      Pds::Epics::EpicsPvCtrlEnum *p = 
        static_cast<Pds::Epics::EpicsPvCtrlEnum *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlEnum2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_CHAR:
    {
      Pds::Epics::EpicsPvCtrlChar *p = 
        static_cast<Pds::Epics::EpicsPvCtrlChar *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlChar2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_LONG:
    {
      Pds::Epics::EpicsPvCtrlLong *p = 
        static_cast<Pds::Epics::EpicsPvCtrlLong *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlLong2str(*p);
    }
    break;
  case Pds::Epics::DBR_CTRL_DOUBLE:
    {
      Pds::Epics::EpicsPvCtrlDouble *p = 
        static_cast<Pds::Epics::EpicsPvCtrlDouble *>(static_cast<void *>(xtc->payload()));
      str << epicsPvCtrlDouble2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_STRING:
    {
      Pds::Epics::EpicsPvTimeString *p = 
        static_cast<Pds::Epics::EpicsPvTimeString *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeString2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_SHORT:
    {
      Pds::Epics::EpicsPvTimeShort *p = 
        static_cast<Pds::Epics::EpicsPvTimeShort *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeShort2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_FLOAT:
    {
      Pds::Epics::EpicsPvTimeFloat *p = 
        static_cast<Pds::Epics::EpicsPvTimeFloat *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeFloat2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_ENUM:
    {
      Pds::Epics::EpicsPvTimeEnum *p = 
        static_cast<Pds::Epics::EpicsPvTimeEnum *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeEnum2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_CHAR:
    {
      Pds::Epics::EpicsPvTimeChar *p = 
        static_cast<Pds::Epics::EpicsPvTimeChar *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeChar2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_LONG:
    {
      Pds::Epics::EpicsPvTimeLong *p = 
        static_cast<Pds::Epics::EpicsPvTimeLong *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeLong2str(*p);
    }
    break;
  case Pds::Epics::DBR_TIME_DOUBLE:
    {
      Pds::Epics::EpicsPvTimeDouble *p = 
        static_cast<Pds::Epics::EpicsPvTimeDouble *>(static_cast<void *>(xtc->payload()));
      str << epicsPvTimeDouble2str(*p);
    }
    break;
  default:
    str << "*unknown dbrType: " << pvHeader->dbrType() << "*";
    break;
  } // switch (dbrType)
  return str.str();
}

  std::string epicsPvHeader2Str(const Pds::Epics::EpicsPvHeader &pv) {
    std::ostringstream str;
    str << obj2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlString2str(const Pds::Epics::EpicsPvCtrlString &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlShort2str(const Pds::Epics::EpicsPvCtrlShort &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlFloat2str(const Pds::Epics::EpicsPvCtrlFloat &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlEnum2str(const Pds::Epics::EpicsPvCtrlEnum &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlChar2str(const Pds::Epics::EpicsPvCtrlChar &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlLong2str(const Pds::Epics::EpicsPvCtrlLong &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvCtrlDouble2str(const Pds::Epics::EpicsPvCtrlDouble &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvCtrlHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeString2str(const Pds::Epics::EpicsPvTimeString &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeShort2str(const Pds::Epics::EpicsPvTimeShort &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeFloat2str(const Pds::Epics::EpicsPvTimeFloat &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeEnum2str(const Pds::Epics::EpicsPvTimeEnum &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeChar2str(const Pds::Epics::EpicsPvTimeChar &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeLong2str(const Pds::Epics::EpicsPvTimeLong &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }

  std::string epicsPvTimeDouble2str(const Pds::Epics::EpicsPvTimeDouble &pv) {
    std::ostringstream str;
    str << obj2str<Pds::Epics::EpicsPvHeader>(pv);
    str << " " << obj2str(pv.dbr());
    str << " " << epicsPvValues2str(pv);
    return str.str();
  }


} // namespace psana_test
