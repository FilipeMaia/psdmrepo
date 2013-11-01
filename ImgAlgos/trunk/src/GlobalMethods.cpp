//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <boost/shared_ptr.hpp>
#include "PSEvt/EventId.h"
#include "PSTime/Time.h"
//#include "psana/Module.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
GlobalMethods::GlobalMethods ()
{
}

//--------------
// Destructor --
//--------------
GlobalMethods::~GlobalMethods ()
{
}

//--------------------

std::string 
stringFromUint(unsigned number, unsigned width, char fillchar)
{
  stringstream ssNum; ssNum << setw(width) << setfill(fillchar) << number;
  return ssNum.str();
}

//--------------------

std::string
stringTimeStamp(PSEvt::Event& evt, std::string fmt) // fmt="%Y%m%dT%H:%M:%S%f"
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return (eventId->time()).asStringFormat(fmt);
  else               return std::string("time-stamp-is-unavailable");
}

//--------------------

double
doubleTime(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return double(eventId->time().sec()) + 1e-9*eventId->time().nsec();
  else               return double(0);
}

//--------------------

std::string  
stringRunNumber(PSEvt::Event& evt, unsigned width)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return stringFromUint(eventId->run(), width);
  else               return std::string("run-is-not-defined");
}

//--------------------

/// Returns integer run number
int 
getRunNumber(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return eventId->run();
  } else {
    MsgLogRoot(warning, "Cannot determine run number, will use 0.");
    return int(0);
  }
}

//--------------------

unsigned 
fiducials(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return eventId->fiducials();
  else               return 0;
}

//--------------------

unsigned 
eventCounterSinceConfigure(PSEvt::Event& evt)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return eventId->vector();
  else               return 0;
}

//--------------------

void 
printSizeOfTypes()
{
  std::cout << "Size Of Types:" 
            << "\nsizeof(bool    ) = " << sizeof(bool    ) << " with typeid(bool    ).name(): " << typeid(bool    ).name() 
            << "\nsizeof(uint8_t ) = " << sizeof(uint8_t ) << " with typeid(uint8_t ).name(): " << typeid(uint8_t ).name()  
            << "\nsizeof(uint16_t) = " << sizeof(uint16_t) << " with typeid(uint16_t).name(): " << typeid(uint16_t).name()  
            << "\nsizeof(int16_t ) = " << sizeof(int16_t ) << " with typeid(int16_t ).name(): " << typeid(int16_t ).name()  
            << "\nsizeof(uint32_t) = " << sizeof(uint32_t) << " with typeid(uint32_t).name(): " << typeid(uint32_t).name()  
            << "\nsizeof(int32_t ) = " << sizeof(int32_t ) << " with typeid(int32_t ).name(): " << typeid(int32_t ).name()  
            << "\nsizeof(int     ) = " << sizeof(int     ) << " with typeid(int     ).name(): " << typeid(int     ).name()  
            << "\nsizeof(float   ) = " << sizeof(float   ) << " with typeid(float   ).name(): " << typeid(float   ).name()  
            << "\nsizeof(double  ) = " << sizeof(double  ) << " with typeid(double  ).name(): " << typeid(double  ).name()  
            << "\nsizeof(short   ) = " << sizeof(short   ) << " with typeid(short   ).name(): " << typeid(short   ).name()  
            << "\n\n";
}

//--------------------
// Define the shape or throw message that can not do that.
void 
defineImageShape(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, unsigned* shape)
{
  if ( defineImageShapeForType<double>  (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<float>   (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<int>     (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<int32_t> (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<uint32_t>(evt, src, key, shape) ) return;
  if ( defineImageShapeForType<uint16_t>(evt, src, key, shape) ) return;
  if ( defineImageShapeForType<uint8_t> (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<int16_t> (evt, src, key, shape) ) return;
  if ( defineImageShapeForType<short>   (evt, src, key, shape) ) return;

  const std::string msg = "Image shape is tested for double, uint16_t, int, float, uint8_t, int16_t, short and is not defined in the event(...)\nfor source:" 
                        + boost::lexical_cast<std::string>(src) + " key:" + key + "\nEXIT psana...";
  MsgLogRoot(error, msg);
  throw std::runtime_error("EXIT psana...");
}

//--------------------

void 
saveTextInFile(const std::string& fname, const std::string& text, bool print_msg)
{
  std::ofstream out(fname.c_str()); 
  out << text;
  out.close();
  //std::setprecision(9); // << std::setw(8) << std::setprecision(0) << std::fixed 

  if( print_msg ) MsgLog("GlobalMethods", info, "Save text in file " << fname.c_str());
}

//--------------------

std::string  
stringInstrument(PSEnv::Env& env)
{
  return env.instrument();
}

//--------------------

std::string  
stringExperiment(PSEnv::Env& env)
{
  return env.experiment();
}

//--------------------

unsigned 
expNum(PSEnv::Env& env)
{
  return env.expNum();
}

//--------------------

std::string
stringExpNum(PSEnv::Env& env, unsigned width)
{
  return stringFromUint(env.expNum(), width);
}

//--------------------

bool 
file_exists(std::string& fname)
{ 
  std::ifstream f(fname.c_str()); 
  return f; 
}

//--------------------

//--------------------

//--------------------
//--------------------
//--------------------
} // namespace ImgAlgos
