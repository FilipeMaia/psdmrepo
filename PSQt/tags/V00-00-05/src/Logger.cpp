//--------------------------

#include "PSQt/Logger.h"
#include <iomanip> // for setw, setfill

using namespace std;   // for cout without std::

namespace PSQt {

//--------------------------

Logger* Logger::p_Logger = NULL; /// make global pointer !!!

//--------------------------

Logger::Logger()
  : QObject(NULL)
  , LoggerBase()
{
  std::cout << "Logger::Logger() - singleton object is created\n";
}

//--------------------------

Logger*
Logger::getLogger()
{
  if( !p_Logger ) p_Logger = new Logger();
  return p_Logger;
}

//--------------------------

void
Logger::new_record(Record& rec)
{
  //std::cout << "Logger::new_record() - is re-implemented\n";
  //std::cout << rec.strRecordTotal() << '\n';  
  // !!!!!!!!!! Emit signal about new record to GUILogger
  emit signal_new_record(rec);
}

//--------------------------

} // namespace PSQt

//--------------------------


