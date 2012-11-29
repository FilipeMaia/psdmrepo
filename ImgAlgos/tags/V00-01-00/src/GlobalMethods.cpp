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
defineImageShape(PSEvt::Event& evt, const PSEvt::Source& str_src, const std::string& str_key, unsigned* shape)
{
  boost::shared_ptr< ndarray<double,2> > img = evt.get(str_src, str_key);
  if (img.get()) {
    for(int i=0;i<2;i++) shape[i]=img->shape()[i];
    //shape=img->shape();
    return;
  } 

  boost::shared_ptr< ndarray<uint16_t,2> > img_u16 = evt.get(str_src, str_key);
  if (img_u16.get()) {
    for(int i=0;i<2;i++) shape[i]=img_u16->shape()[i];
    return;
  } 

  boost::shared_ptr< ndarray<int,2> > img_int = evt.get(str_src, str_key);
  if (img_int.get()) {
    for(int i=0;i<2;i++) shape[i]=img_int->shape()[i];
    return;
  } 

  boost::shared_ptr< ndarray<float,2> > img_flo = evt.get(str_src, str_key);
  if (img_flo.get()) {
    for(int i=0;i<2;i++) shape[i]=img_flo->shape()[i];
    return;
  } 

  boost::shared_ptr< ndarray<uint8_t,2> > img_u8 = evt.get(str_src, str_key);
  if (img_u8.get()) {
    for(int i=0;i<2;i++) shape[i]=img_u8->shape()[i];
    return;
  } 

  const std::string msg = "Image shape is tested for double, uint16_t, int, float, uint8_t and is not defined in the event(...)\nfor source:" 
                        + boost::lexical_cast<std::string>(str_src) + " key:" + str_key;
  //MsgLogRoot(error, msg);
  throw std::runtime_error(msg);

}

//--------------------
//--------------------
} // namespace ImgAlgos
