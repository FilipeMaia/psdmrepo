//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpDgram
//
// Author List:
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpDgram.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpDgram)

namespace {
    

} // local namespace

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpDgram::DumpDgram (const std::string& name)
  : Module(name)
{
}

//--------------
// Destructor --
//--------------
DumpDgram::~DumpDgram ()
{
}

void 
DumpDgram::beginJob(Event& evt, Env& env)
{
  dgramDump(evt,"beginJob()");
}

void 
DumpDgram::beginRun(Event& evt, Env& env)
{
  dgramDump(evt, "beginRun()");
}

void 
DumpDgram::beginCalibCycle(Event& evt, Env& env)
{
  dgramDump(evt, "beginCalibCycle()");
}

void 
DumpDgram::event(Event& evt, Env& env)
{
  dgramDump(evt, "event()");
}

/// Method which is called at the end of the calibration cycle
void
DumpDgram::endCalibCycle(Event& evt, Env& env)
{
  dgramDump(evt, "endCalibCycle()");
}

/// Method which is called at the end of the run
void
DumpDgram::endRun(Event& evt, Env& env)
{
  dgramDump(evt, "endRun()");
}

/// Method which is called once at the end of the job
void
DumpDgram::endJob(Event& evt, Env& env)
{
  dgramDump(evt, "endJob()");
}

void
DumpDgram::dgramDump(Event &evt, const std::string & hdr) 
{
  XtcInput::Dgram::ptr dgptr = evt.get();
  boost::shared_ptr<XtcInput::DgramList> dgListptr = evt.get();
  if ((not dgptr) and (not dgListptr)) {
    MsgLog( name(), info, " " << hdr << ": no XtcInput::DgramList or Pds::Dgram found in Event.");
  }
  if (dgptr) {
    const Pds::ClockTime &clock = dgptr->seq.clock();
    const Pds::TimeStamp & stamp = dgptr->seq.stamp();
    MsgLog(name(), info, " " << hdr << ": Pds::Dgram found: "
           << " sec=" << clock.seconds()
           << " nsec=" << clock.nanoseconds()
           << " fid=" << stamp.fiducials());
  }
  if (dgListptr) {
    std::vector<XtcInput::Dgram::ptr> dgrams=  dgListptr->getDgrams();
    if (dgrams.size()==0) {
      MsgLog( name(), info, " " << hdr << ": XtcInput::DgramList found with no dgrams.");
      return;
    }
    MsgLog( name(), info, " " << hdr << ": XtcInput::DgramList found with " << dgrams.size() << " dgrams.");
    std::vector<XtcInput::XtcFileName> files =  dgListptr->getFileNames();
    std::vector<off64_t> offsets =  dgListptr->getOffsets();
    for (unsigned idx = 0; idx < files.size(); ++idx) {
      XtcInput::XtcFileName &file = files.at(idx);
      off64_t offset = offsets.at(idx);
      XtcInput::Dgram::ptr &dgptr = dgrams.at(idx);
      const Pds::ClockTime &clock = dgptr->seq.clock();
      const Pds::TimeStamp & stamp = dgptr->seq.stamp();
      std::cout << " dg " << idx << ": " 
                << " sec=" << clock.seconds()
                << " nsec=" << clock.nanoseconds()
                << " fid=" << stamp.fiducials()
                << " basename=" << file.basename()
                << " run=" << file.run()
                << " stream=" << file.stream()
                << " chunk=" << file.chunk()
		<< " offset=" << offset
                << std::endl;
    }
  }
}

} // namespace psana_examples
