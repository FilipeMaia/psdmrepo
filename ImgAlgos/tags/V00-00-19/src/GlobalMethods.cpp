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
#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream

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
// Define the shape or throw message that can not do that.
void 
defineImageShape(PSEvt::Event& evt, const std::string& m_str_src, const std::string& m_key, unsigned* shape)
{
  boost::shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key);
  if (img.get()) {
    for(int i=0;i<2;i++) shape[i]=img->shape()[i];
    //shape=img->shape();
  } 
  else
  {
    const std::string msg = "Image shape is not defined in the event(...) for source:" + m_str_src + " key:" + m_key;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }
}

//--------------------
//--------------------
} // namespace ImgAlgos
