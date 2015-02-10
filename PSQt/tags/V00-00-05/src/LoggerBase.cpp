//--------------------------

#include "PSQt/LoggerBase.h"
#include <iomanip> // for setw, setfill

using namespace std;   // for cout without std::

namespace PSQt {

//--------------------------

LoggerBase::LoggerBase()
  : m_recnum(0)
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
LoggerBase::message(const std::string& name, const LEVEL& level, const std::string& msg)
{
  Record rec = {++m_recnum, doubleTimeNow(), strTimeStamp(), level, name, msg};
  m_sslog[m_recnum] = rec;
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
  stringstream ss; ss << strTimeStamp() << " LoggerBase::print(): " << str;
  std::cout << ss.str() << '\n';
}

//--------------------------

std::string 
LoggerBase::strRecordsForLevel(const LEVEL& threshold)
{
  stringstream ss;
  for (std::map<unsigned, Record>::iterator it=m_sslog.begin(); it!=m_sslog.end(); ++it) {
    Record& rec = it->second; 
    if(rec.level>=threshold) ss << rec.strRecord() << '\n';
  }

  return ss.str();
}

//--------------------------

void
LoggerBase::saveLogInFile(const std::string& ofname)
{
  std::string fname = (ofname==std::string()) ? m_start_tstamp+"-log.txt" : ofname;

  // open file
  std::ofstream out(fname.c_str());
  if (not out.good()) { std::cout << "File " << fname << " IS NOT SAVED!\n"; return; }
  
  for (std::map<unsigned, Record>::iterator it=m_sslog.begin(); it!=m_sslog.end(); ++it)
    out << (it->second).strRecordTotal() << '\n';

  out.close();
}

//--------------------------

void
LoggerBase::new_record(Record& rec)
{
  if(rec.number < 3) std::cout << "LoggerBase::new_record() - DEFAULT callback for re-implementation in subclass\n";
  std::cout << rec.strRecordTotal() << '\n';
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
     << std::setw(10) << name << ' '
     << std::setw(10) << strLevel(level)
     << msg;
  return ss.str();
}

//--------------------------

std::string 
Record::strRecord()
{
  stringstream ss; 
  ss << left << std::setw(21) << tstamp
     << std::setw(10) << name << ' '
     << std::setw(10) << strLevel(level)
     << msg;
  return ss.str();
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
  else if(level == ERROR   ) return "ERROR";
  else if(level == CRITICAL) return "CRITICAL";
  else                       return "NON-IMPLEMENTED";
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


