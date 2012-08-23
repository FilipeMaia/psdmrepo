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
#include <sstream>   // for streamstring

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

std::string  
stringRunNumber(PSEvt::Event& evt, unsigned width)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) return stringFromUint(eventId->run(), width);
  else               return std::string("run-is-not-defined");
}

//--------------------
//--------------------

//--------------------
} // namespace ImgAlgos
