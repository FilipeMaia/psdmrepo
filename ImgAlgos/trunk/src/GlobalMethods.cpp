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
#include <fstream>   // ofstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <boost/shared_ptr.hpp>
#include "MsgLogger/MsgLogger.h"
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
// Define the shape or throw message that can not do that.
void 
defineImageShape(PSEvt::Event& evt, const std::string& m_str_src, const std::string& m_key, unsigned* shape)
{
  boost::shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key);
  if (img.get()) {
    for(int i=0;i<2;i++) shape[i]=img->shape()[i];
  } 
  else
  {
    const std::string msg = "Image shape is not defined in the event(...) for source:" + m_str_src + " key:" + m_key;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }
}

//--------------------
// Save 2-D array in file
void 
save2DArrInFile(const std::string& fname, const double* arr, const unsigned& rows, const unsigned& cols, bool print_msg )
{  
  if (fname.empty()) {
    MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
    return;
  }

  if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str());
  std::ofstream out(fname.c_str());
        for (unsigned r = 0; r != rows; ++r) {
          for (unsigned c = 0; c != cols; ++c) {
            out << arr[r*cols + c] << ' ';
          }
          out << '\n';
        }
  out.close();
}

//--------------------
//--------------------
} // namespace ImgAlgos
