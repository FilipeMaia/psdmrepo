//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadBaseModule...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadBaseModule.h"

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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadBaseModule::CSPadBaseModule (const std::string& name,
    const std::string& keyName,
    const std::string& defKey,
    const std::string& sourceName,
    const std::string& defSource)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_src()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_key     = configStr(keyName, defKey);
  m_str_src = configSrc(sourceName, defSource);

  // initialize arrays
  std::fill_n(&m_segMask[0], int(Psana::CsPad::MaxQuadsPerSensor), 0U);
}

//--------------
// Destructor --
//--------------
CSPadBaseModule::~CSPadBaseModule ()
{
}

/// Method which is called at the beginning of the run
void 
CSPadBaseModule::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration
  // object is found then complain and stop.

  int count = 0;

  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(m_str_src, &m_src);
  if (config1.get()) {
    MsgLog(name(), debug, "Found CsPad::ConfigV1 object with address " << m_src);
    for (int i = 0; i < Psana::CsPad::MaxQuadsPerSensor; ++i) { m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff; }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_str_src, &m_src);
  if (config2.get()) {
    MsgLog(name(), debug, "Found CsPad::ConfigV2 object with address " << m_src);
    for (int i = 0; i < Psana::CsPad::MaxQuadsPerSensor; ++i) { m_segMask[i] = config2->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(m_str_src, &m_src);
  if (config3.get()) {
    MsgLog(name(), debug, "Found CsPad::ConfigV3 object with address " << m_src);
    for (int i = 0; i < Psana::CsPad::MaxQuadsPerSensor; ++i) { m_segMask[i] = config3->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV4> config4 = env.configStore().get(m_str_src, &m_src);
  if (config4.get()) {
    MsgLog(name(), debug, "Found CsPad::ConfigV4 object with address " << m_src);
    for (int i = 0; i < Psana::CsPad::MaxQuadsPerSensor; ++i) { m_segMask[i] = config4->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV5> config5 = env.configStore().get(m_str_src, &m_src);
  if (config5.get()) {
    MsgLog(name(), debug, "Found CsPad::ConfigV5 object with address " << m_src);
    for (int i = 0; i < Psana::CsPad::MaxQuadsPerSensor; ++i) { m_segMask[i] = config5->roiMask(i); }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CSPad configuration objects found. Terminating.");
    terminate();
    return;
  }

  if (count > 1) {
    MsgLog(name(), error, "Multiple CSPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CSPad object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found CSPad configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed CSPad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found CSPad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}

/// Methods for event processing interface

void
CSPadBaseModule::initData() {}

void 
CSPadBaseModule::procQuad(unsigned quad, const int16_t* data) {}

void
CSPadBaseModule::summaryData(Event& evt) {}

//--------------------
// Print base parameters
void 
CSPadBaseModule::printBaseParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Base parameters:"
        << "\n MaxQuads   : " << MaxQuads    
        << "\n MaxSectors : " << MaxSectors  
        << "\n NumColumns : " << NumColumns  
        << "\n NumRows    : " << NumRows     
        << "\n SectorSize : " << SectorSize  
        << "\n"
        << "\n Input parameters:"
        << "\n source     : " << sourceConfigured()
        << "\n key        : " << inputKey()
        << "\n";
  }
}

} // namespace ImgAlgos
