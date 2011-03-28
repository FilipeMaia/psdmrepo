//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEpics...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEpics.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEpics)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpEpics"; 
 
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEpics::DumpEpics (const std::string& name)
  : Module(name)
{
}

//--------------
// Destructor --
//--------------
DumpEpics::~DumpEpics ()
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpEpics::event(Event& evt, Env& env)
{
  // access EPICS store
  const EpicsStore& estore = env.epicsStore();
  
  // get the names of EPICS PVs
  std::vector<std::string> pvNames = estore.pvNames();

  size_t size = pvNames.size();
  std::cout  << "Total number of EPICS PVs: " << pvNames.size() << '\n';
  
  for (size_t i = 0; i < size; ++ i) {
    
    // get generic PV object, only useful if you want to access
    // its type, and array size
    shared_ptr<Psana::Epics::EpicsPvHeader> pv = estore.getPV(pvNames[i]);

    // print generic info
    std::cout << "  " << i << ". " << pvNames[i] << " id=" << pv->pvId() 
              << " type=" << pv->dbrType() << " size=" << pv->numElements() << '\n';

    // print status info
    int status, severity;
    PSTime::Time time;
    estore.status(pvNames[i], status, severity, time);
    std::cout << "    status=" << status << ", severity=" << severity 
              << " time=" << time << '\n';

    // print all values
    std::cout << "    values:";
    for (int e = 0; e < pv->numElements(); ++ e) {
      // get value and convert to string
      const std::string& value = estore.value(pvNames[i], e);
      std::cout << ' ' << value;
    }
    std::cout << '\n';

  }
  
  
}

} // namespace psana_examples
