//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrSaveInFile...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrSaveInFile.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream
//#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(CSPadArrSaveInFile)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadArrSaveInFile::CSPadArrSaveInFile (const std::string& name)
  : CSPadBaseModule(name)
  , m_outFile()
  , m_print_bits()
{
  m_outFile     = configStr("outfile", "cspad-arr"); // ".txt"
  m_print_bits  = config("print_bits",  0);
}

//--------------
// Destructor --
//--------------
CSPadArrSaveInFile::~CSPadArrSaveInFile ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadArrSaveInFile::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrSaveInFile::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrSaveInFile::event(Event& evt, Env& env)
{
   if ( procEventForType<Psana::CsPad::DataV1, CsPad::ElementV1> (evt) ) return;
   if ( procEventForType<Psana::CsPad::DataV2, CsPad::ElementV2> (evt) ) return;

   MsgLog(name(), warning, "event(...): Psana::CsPad::DataV# / ElementV# is not available in this event.");
}


/// Method which is called at the end of the calibration cycle
void 
CSPadArrSaveInFile::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrSaveInFile::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrSaveInFile::endJob(Event& evt, Env& env)
{
}

//--------------------

/// Process statistics for quad; combine it in a single 4-d array
void 
CSPadArrSaveInFile::procQuad(unsigned quad, const int16_t* data)
{
  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
      // beginning of the segment data
      int16_t*       arr     = &m_arr [quad][sect][0][0];
      const int16_t* segData = data + ind_in_arr*SectorSize;

      for (int i = 0; i < SectorSize; ++ i) {
	arr[i] = segData[i];
      }          
      
      ++ind_in_arr;
    }
  }
}

//--------------------
/// Implementation for abstract method from CSPadBaseModule.h
// Prints something, get the file name, and saves the CSPad array in file 
void 
CSPadArrSaveInFile::summaryData(Event& evt)
{
  saveInFile(evt);
}

//--------------------
// Prints something, get the file name, and saves the CSPad array in file 
void 
CSPadArrSaveInFile::saveInFile(Event& evt)
{
  if( m_print_bits & 2 ) printEventId(evt);
  if( m_print_bits & 4 ) printTimeStamp(evt);

  std::string fname = strTimeDependentFileName(evt);
  saveCSPadArrayInFile<int16_t>(fname, m_arr); // or &m_arr[0][0][0][0];
}

//--------------------
/// Save 4-d array of CSPad structure in file
template <typename T>
void 
CSPadArrSaveInFile::saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows])
{  
  if (not fname.empty()) {
    if( m_print_bits & 8 ) MsgLog(name(), info, "Save CSPad-shaped array in file " << fname.c_str());
    std::ofstream out(fname.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            out << arr[iq][is][ic][ir] << ' ';
          }
          out << '\n';
        }
      }
    }
    out.close();
  }
}

//--------------------

// Print input parameters
void 
CSPadArrSaveInFile::printInputParameters()
{
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : " << sourceConfigured()
        << "\n key        : " << inputKey()
        << "\n m_outFile  : " << m_outFile    
        << "\n print_bits : " << m_print_bits
        << "\n";     
  }
}

//--------------------

void 
CSPadArrSaveInFile::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << counter() << " ID: " << *eventId);
  }
}

//--------------------

void 
CSPadArrSaveInFile::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, "Run="    <<  eventId->run()
                       << " Event=" <<  counter() 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------

std::string
CSPadArrSaveInFile::strEventCounter()
{
  std::stringstream ssEvNum; ssEvNum << std::setw(6) << std::setfill('0') << counter();
  return ssEvNum.str();
}

//--------------------

std::string  
CSPadArrSaveInFile::strTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    return (eventId->time()).asStringFormat( "%Y-%m-%d-%H%M%S%f"); // "%Y-%m-%d %H:%M:%S%f%z"
  }
  else
    return std::string("time-stamp-is-not-defined");
}

//--------------------

std::string  
CSPadArrSaveInFile::strRunNumber(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    std::stringstream ssRunNum; ssRunNum << "r" << std::setw(4) << std::setfill('0') << eventId->run();
    return ssRunNum.str();
  }
  else
    return std::string("run-is-not-defined");
}

//--------------------

std::string
CSPadArrSaveInFile::strTimeDependentFileName(Event& evt)
{
  // Define the file name
  std::string fname = m_outFile 
                    + "-" + strEventCounter() 
                    + "-" + strRunNumber(evt) 
                    + "-" + strTimeStamp(evt) + ".txt";
  return fname;
}

//--------------------

} // namespace ImgAlgos
