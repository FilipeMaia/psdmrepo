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
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "psddl_hdf2psana/HdfGroupName.h"
#include "psddl_hdf2psana/Exceptions.h"
#include "psddl_hdf2psana/dispatch.h"

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
  const std::string epicsGroupName("Epics::EpicsPv");

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
HdfConverter::convert(const hdf5pp::Group& group, uint64_t idx, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
{
  const std::string& typeName = this->typeName(group);
  const Pds::Src& src = this->source(group);
  int schema = schemaVersion(group);
  
  hdfConvert(group, idx, typeName, schema, src, &evt, cfgStore);
}

/**
 *  @brief Convert one object and store it in the config store.
 */
void
HdfConverter::convertConfig(const hdf5pp::Group& group, uint64_t idx, PSEnv::EnvObjectStore& cfgStore)
{
  const std::string& typeName = this->typeName(group);
  const Pds::Src& src = this->source(group);
  int schema = schemaVersion(group);
  
  hdfConvert(group, idx, typeName, schema, src, 0, cfgStore);
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void
HdfConverter::convertEpics(const hdf5pp::Group& group, uint64_t idx, PSEnv::EpicsStore& eStore)
{
}

/**
 *  @brief This method should be called to reset cache whenever some groups are closed
 */
void
HdfConverter::resetCache()
{
  m_schemaVersionCache.clear();
  m_isEpicsCache.clear();
  m_typeNameCache.clear();
  m_sourceCache.clear();
}

bool
HdfConverter::isEpics(const hdf5pp::Group& group, int levels) const
{
  // check cache first
  std::map<hdf5pp::Group, bool>::const_iterator it = m_isEpicsCache.find(group);
  if (it !=  m_isEpicsCache.end()) return it->second;

  // look at group name
  bool res = group.basename() == ::epicsGroupName;
  if (not res and levels > 0) {
    // try its parent
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) res = isEpics(parent, levels - 1);
  }

  // update cache
  m_isEpicsCache.insert(std::make_pair(group, res));

  return res;
}

int
HdfConverter::schemaVersion(const hdf5pp::Group& group, int levels) const
{
  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return schemaVersion(group, isEpics(group) ? 2 : 1);

  // check cache first
  std::map<hdf5pp::Group, int>::const_iterator it = m_schemaVersionCache.find(group);
  if (it !=  m_schemaVersionCache.end()) return it->second;

  // look at attribute
  int version = 0;
  hdf5pp::Attribute<int> attr = group.openAttr<int>(::versionAttrName);
  if (attr.valid()) {
    version = attr.read();
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) version = schemaVersion(parent, levels - 1);
  }

  // update cache
  m_schemaVersionCache.insert(std::make_pair(group, version));

  return version;
}

// Get TypeId for the group or its parent (and its grand-parent for EPICS),
std::string
HdfConverter::typeName(const hdf5pp::Group& group, int levels) const
{
  std::string typeName;
  
  // check cache first
  std::map<hdf5pp::Group, std::string>::const_iterator it = m_typeNameCache.find(group);
  if (it !=  m_typeNameCache.end()) {
    typeName = it->second;
  } else if (levels < 0) {
    // with default argument call myself with correct level depending on type of group
    typeName = this->typeName(group, isEpics(group) ? 2 : 1);
  } else if (levels == 0) {
    // type name is a group name for top-level type group
    typeName = group.basename();
    // update cache
    m_typeNameCache.insert(std::make_pair(group, typeName));
  } else {
    // try parent group
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) typeName = this->typeName(parent, levels - 1);
  }

  return typeName;
}


// Get Source for the group (or its parent for EPICS),
Pds::Src
HdfConverter::source(const hdf5pp::Group& group, int levels) const
{
  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return source(group, isEpics(group) ? 1 : 0);

  // check cache first
  std::map<hdf5pp::Group, Pds::Src>::const_iterator it = m_sourceCache.find(group);
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
  m_sourceCache.insert(std::make_pair(group, src));

  return src;
}


} // namespace psddl_hdf2psana
