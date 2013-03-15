//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfConverter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/HdfConverter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Utils.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "psddl_hdf2psana/HdfGroupName.h"
#include "psddl_hdf2psana/Exceptions.h"
#include "psddl_hdf2psana/dispatch.h"
#include "psddl_hdf2psana/epics.ddlm.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const std::string logger = "HdfConverter";

  // name of the attribute holding schema version
  const std::string versionAttrName("_psddlSchemaVersion");

  // name of the attributes holding Src info
  const std::string srcAttrName("_xtcSrc");

  // name of the group holding EPICS data
  const std::string epicsGroupName("/Epics::EpicsPv/");

  // helper class to build Src from stored 64-bit code
  class _SrcBuilder : public Pds::Src {
  public:
    _SrcBuilder(uint64_t value) {
      _phy = uint32_t(value >> 32);
      _log = uint32_t(value);
    }
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_hdf2psana {

//----------------
// Constructors --
//----------------
HdfConverter::HdfConverter ()
{
}

//--------------
// Destructor --
//--------------
HdfConverter::~HdfConverter ()
{
}

/**
 *  @brief Convert one object and store it in the event.
 */
void
HdfConverter::convert(const hdf5pp::Group& group, uint64_t idx, PSEvt::Event& evt, PSEnv::Env& env)
{
  const std::string& typeName = this->typeName(group.name());
  const Pds::Src& src = this->source(group);
  int schema = schemaVersion(group);
  
  hdfConvert(group, idx, typeName, schema, src, evt, env.configStore());
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void
HdfConverter::convertEpics(const hdf5pp::Group& group, uint64_t idx, PSEnv::EpicsStore& eStore)
{
  if (isEpics(group.name())) {
    boost::shared_ptr<Psana::Epics::EpicsPvHeader> epics = Epics::readEpics(group, idx);
    if (epics) {
      const Pds::Src& src = this->source(group);
      eStore.store(epics, src);
    }
  }
}

/**
 *  @brief This method should be called to reset cache whenever some groups are closed
 */
void
HdfConverter::resetCache()
{
  MsgLog(logger, debug, "HdfConverter::resetCache");
  m_schemaVersionCache.clear();
  m_sourceCache.clear();
}

// test if the group is inside EPICS group (has a parent named Epics::EpicsPv)
bool
HdfConverter::isEpics(const std::string& group) const
{
  MsgLog(logger, debug, "HdfConverter::isEpics - group: " << group);

  // look at group name
  bool res = group.find(::epicsGroupName) != std::string::npos;

  return res;
}

int
HdfConverter::schemaVersion(const hdf5pp::Group& group, int levels) const
{
  MsgLog(logger, debug, "HdfConverter::schemaVersion - group: " << group);

  const std::string& name = group.name();
  int version = 0;

  // with default argument call myself with correct level depending on type of group
  if (levels < 0) {

    // check cache first
    std::map<std::string, int>::const_iterator it = m_schemaVersionCache.find(name);
    if (it !=  m_schemaVersionCache.end()) return it->second;

    version = schemaVersion(group, isEpics(name) ? 2 : 1);

    // update cache
    m_schemaVersionCache.insert(std::make_pair(name, version));

    return version;
  }

  // look at attribute
  hdf5pp::Attribute<int> attr = group.openAttr<int>(::versionAttrName);
  if (attr.valid()) {
    version = attr.read();
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) version = schemaVersion(parent, levels - 1);
  }

  return version;
}

// Get TypeId for the group or its parent (and its grand-parent for EPICS),
std::string
HdfConverter::typeName(const std::string& group) const
{
  MsgLog(logger, debug, "HdfConverter::typeName - group: " << group);

  std::string typeName = group;
  std::string::size_type p = typeName.rfind('/');
  typeName.erase(p);
  if (isEpics(group)) {
    p = typeName.rfind('/');
    typeName.erase(p);
  }
  p = typeName.rfind('/');
  typeName.erase(0, p+1);

  return typeName;
}


// Get Source for the group (or its parent for EPICS),
Pds::Src
HdfConverter::source(const hdf5pp::Group& group, int levels) const
{
  MsgLog(logger, debug, "HdfConverter::source - group: " << group);

  const std::string& name = group.name();

  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return source(group, isEpics(name) ? 1 : 0);

  // check cache first
  std::map<std::string, Pds::Src>::const_iterator it = m_sourceCache.find(name);
  if (it !=  m_sourceCache.end()) return it->second;

  // look at attribute
  Pds::Src src(Pds::Level::NumberOfLevels);
  hdf5pp::Attribute<uint64_t> attrSrc = group.openAttr<uint64_t>(::srcAttrName);
  if (attrSrc.valid()) {
    // build TypeId from attributes
    src = ::_SrcBuilder(attrSrc.read());
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) src = this->source(parent, levels - 1);
  } else if (levels == 0) {
    // guess type id from group name for top-level type group
    src = HdfGroupName::nameToSource(group.basename());

    // some corrections needed for incorrectly stored names
    if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 0)) {
      src = Pds::BldInfo(0, Pds::BldInfo::EBeam);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 1)) {
      src = Pds::BldInfo(0, Pds::BldInfo::PhaseCavity);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 2)) {
      src = Pds::BldInfo(0, Pds::BldInfo::FEEGasDetEnergy);
    }
  }

  // update cache
  m_sourceCache.insert(std::make_pair(name, src));

  return src;
}


} // namespace psddl_hdf2psana
