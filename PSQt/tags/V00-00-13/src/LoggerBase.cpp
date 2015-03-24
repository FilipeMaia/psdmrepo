//--------------------------

#include "PSQt/LoggerBase.h"
#include "PSQt/QGUtils.h"
#include <iomanip> // for setw, setfill

using namespace std;   // for cout without std::

namespace PSQt {

//--------------------------

LoggerBase::LoggerBase(const LEVEL& level)
  : m_recnum(0)
  , m_level(level)
{
  startLoggerBase();
}

//--------------------------

void 
LoggerBase::startLoggerBase()
{
  m_start_tstamp = strTimeStamp();
  m_start_time   = doubleTimeNow();
  stringstream ss; ss << "Start logger at t[sec] = " << fixed << std::setprecision(9) << m_start_time;
  message(_name_(), INFO, ss.str());
}

//--------------------------

void 
LoggerBase::setLevel(const LEVEL& level) 
{ 
  m_level=level; 
  message(_name_(), DEBUG, "Message level is set to " + strLevel(level));
}

//--------------------------

void
LoggerBase::message(const std::string& name, const LEVEL& level, const std::string& msg)
{
  // All records are saved in the log
  Record rec = {++m_recnum, doubleTimeNow(), strTimeStamp(), level, name, msg};
  m_sslog[m_recnum] = rec;

  // Check input message level. Messages with level lover than threshold are ignored in callback.
  if(level < m_level) return;

  if(m_level == DEBUG) std::cout << rec.strRecordTotal() << '\n';
  new_record(rec);
}

//--------------------------

void
LoggerBase::message(const std::string& name, const LEVEL& level, const std::stringstream& ss)
{
  message(name, level, ss.str());
}

//--------------------------

void
LoggerBase::print(const std::string& str)
{
  stringstream ss; ss << strTimeStamp() << " LoggerBase::print():" << str;
  std::cout << ss.str() << '\n';
}

//--------------------------

std::string 
LoggerBase::strRecordsForLevel(const LEVEL& threshold)
{
  stringstream ss;
  for (std::map<unsigned, Record>::iterator it=m_sslog.begin(); it!=m_sslog.end(); ++it) {
    Record& rec = it->second; 
    if(rec.level<threshold) continue;
    ss << rec.strRecord() << '\n';
  }
  return ss.str();
}

//--------------------------

void
LoggerBase::saveLogInFile(const std::string& ofname, const bool& add_tstamp)
{
  std::string fname = getFileName(ofname, add_tstamp);

  message(_name_(), INFO, "Save this log in the file " + fname);

  // open file
  std::ofstream out(fname.c_str());
  if (not out.good()) { message(_name_(), WARNING, "File " + fname + " IS NOT SAVED! - status is not good"); return; }
  
  out << this->strRecordsForLevel(m_level);

  out.close();

  print(" Log saved in the file " + fname);
}

//--------------------------

std::string
LoggerBase::getFileName(const std::string& ofname, const bool& add_tstamp)
{
  std::string fname = (ofname==std::string()) ? "log.txt" : ofname;

  if (! add_tstamp) return fname;

  std::string root;
  std::string ext;
  splitext(fname, root, ext);
  return root + '-' + m_start_tstamp + ext; 
}

//--------------------------

void
LoggerBase::new_record(Record& rec)
{
  if(rec.number < 2) message(_name_(), DEBUG, "new_record() - DEFAULT version of callback method for overloading in subclass");
  //  print(_name_() + "new_record() - DEFAULT version of callback method for overloading in subclass");

  //std::cout << rec.strRecordTotal() << '\n';
}

//--------------------------
//--------------------------
//-- Methods of Record  ----
//--------------------------
//--------------------------

std::string 
Record::strRecordTotal()
{
  stringstream ss;
  ss << left << std::setw(4) << number
     << fixed << std::setw(16) << std::setprecision(3) << time 
     << std::setw(21) << tstamp
     << std::setw(12) << name << ' '
     << std::setw(10) << strLevel(level)
     << msg;
  return ss.str();
}

//--------------------------

std::string
Record::strRecordBrief()
{
  stringstream ss; 
  ss << left << std::setw(4) << number
     << std::setw(21) << tstamp
     << std::setw(10) << name << ' '
     << std::setw(10) << strLevel(level)
     << msg;
  return ss.str();
}

//--------------------------

std::string
Record::strRecord()
{
  //return strRecordTotal();
  return strRecordBrief();
}

//--------------------------
//--------------------------
//--------------------------
//---  Global methods  -----
//--------------------------
//--------------------------
//--------------------------

std::string
strLevel(const LEVEL& level)
{
  if     (level == DEBUG   ) return "DEBUG";
  else if(level == INFO    ) return "INFO";
  else if(level == WARNING ) return "WARNING";
  else if(level == CRITICAL) return "CRITICAL";
  else if(level == ERROR   ) return "ERROR";
  else                       return "NON-IMPLEMENTED";
}
//--------------------------

LEVEL
levelFromString(const std::string& str)
{
  if     (str == std::string("DEBUG")   ) return DEBUG;
  else if(str == std::string("INFO")    ) return INFO;
  else if(str == std::string("WARNING") ) return WARNING;
  else if(str == std::string("CRITICAL")) return CRITICAL;
  else if(str == std::string("ERROR")   ) return ERROR;
  else                                    return NONE;
}

//--------------------------

std::string strTimeStamp(const std::string& format)
{
  time_t  time_sec;
  time ( &time_sec );
  struct tm* timeinfo; timeinfo = localtime ( &time_sec );
  char c_time_buf[32]; strftime(c_time_buf, 32, format.c_str(), timeinfo);
  return std::string (c_time_buf);
}

//--------------------------

double
doubleTimeNow() {
  struct timespec tnow;
  int status = clock_gettime( CLOCK_REALTIME, &tnow );
  return (status) ? 0 : tnow.tv_sec + 1e-9 * tnow.tv_nsec;
}

//--------------------------

} // namespace PSQt

//--------------------------


