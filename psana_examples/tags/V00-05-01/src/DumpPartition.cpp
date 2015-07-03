//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id:
//
// Description:
//	Class DumpPartition...
//
// Author List:
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpPartition.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/partition.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpPartition)

namespace {
  void printPartition(std::ostream &out, boost::shared_ptr<Psana::Partition::ConfigV1> config) {
    out << "Partition::ConfigV1:\n";
    out << "  bldMask: " <<  "0x" << std::setw(16) << std::setfill('0') << std::hex << config->bldMask() << '\n';
    out << "  numSources: " << config->numSources() << '\n';
    ndarray<const Psana::Partition::Source, 1> sources = config->sources();
    for (unsigned idx = 0; idx < config->numSources(); ++ idx) {
      const Psana::Partition::Source & source = sources[idx];
      const Pds::Src& src = source.src();
      uint32_t group = source.group();
      out << "    src= " << src << "  group= " << group << '\n';
    }
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpPartition::DumpPartition (const std::string& name)
  : Module(name)
{
}

//--------------
// Destructor --
//--------------
DumpPartition::~DumpPartition ()
{
}

void 
DumpPartition::beginJob(Event& evt, Env& env)
{
  PSEvt::Source src("ProcInfo()");
  boost::shared_ptr<Psana::Partition::ConfigV1> config = env.configStore().get(src);
  if (config) {
    MsgLog(name(),info,"DumpPartition In beginJob()");
    printPartition(std::cout, config);
  }
}

// Method which is called with event data
void 
DumpPartition::event(Event& evt, Env& env)
{
}
  
} // namespace psana_examples
