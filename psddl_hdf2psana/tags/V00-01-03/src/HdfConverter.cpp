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
#include <boost/algorithm/string.hpp>

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
#include "psddl_hdf2psana/bld.ddl.h"
#include "psddl_hdf2psana/epics.ddlm.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const std::string logger = "psddl_hdf2psana.HdfConverter";

  // name of the attribute holding schema version
  const std::string versionAttrName("_psddlSchemaVersion");

  // name of the attributes holding Src info
  const std::string srcAttrName("_xtcSrc");

  // name of the group holding EPICS data
  const std::string epicsGroupName("/Epics::EpicsPv/");

  // test if the group is inside EPICS group (has a parent named Epics::EpicsPv)
  bool isEpics(const std::string& group)
  {
    // look at group name
    bool res = group.find(::epicsGroupName) != std::string::npos;
    return res;
  }

  // test if the group is CTRL EPICS group, CTRL EPICS is located right in the Configure group,
  // so its name will start with either /Configure/Epics::EpicsPv/ or /Configure:NNNN/Epics::EpicsPv/
  bool isCtrlEpics(const std::string& group)
  {
    MsgLog(logger, debug, "HdfConverter::isCtrlEpics - group: " << group);

    if (boost::starts_with(group, "/Configure:") or boost::starts_with(group, "/Configure/")) {
      std::string::size_type p = group.find('/', 1);
      if (p != std::string::npos) {
        if (group.compare(p, ::epicsGroupName.size(), ::epicsGroupName) == 0) {
          MsgLog(logger, debug, "HdfConverter::isCtrlEpics - yes");
          return true;
        }
      }
    }

    return false;
  }

  // helper class to build Src from stored 64-bit code
  class _SrcBuilder : public Pds::Src {
  public:
    _SrcBuilder(uint64_t value) {
      _phy = uint32_t(value >> 32);
      _log = uint32_t(value);
    }
  };

  
  // Special proxy which returns sub-object of the larger object 
  template <typename Parent, typename SubObject, const SubObject& (Parent::*Method)() const>
  class SubObjectProxy : public PSEvt::Proxy<SubObject> {
  public:
    SubObjectProxy(const boost::shared_ptr<PSEvt::Proxy<Parent> >& parentProxy) : m_parent(parentProxy) {}
    
    virtual boost::shared_ptr<SubObject> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key) {
      boost::shared_ptr<SubObject> res;
      if (boost::shared_ptr<void> vptr = m_parent->get(dict, source, key)) {
        boost::shared_ptr<Parent> parent = boost::static_pointer_cast<Parent>(vptr);
        const SubObject& subref = (parent.get()->*Method)();
        res = boost::shared_ptr<SubObject>(parent, const_cast<SubObject*>(&subref));
      }
      return res;
    }
    
  private:
    boost::shared_ptr<PSEvt::Proxy<Parent> > m_parent;
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
  
  /*
   * Special case for Shared BLD data. We split them into their individual
   * components and store them as regular objects instead of one large
   * composite object. Components include both configuration and event data
   * objects so we update config store here as well.
   */
  if (typeName == "Bld::BldDataIpimbV0") {

    const boost::shared_ptr<PSEvt::Proxy<Psana::Bld::BldDataIpimbV0> >& parent = Bld::make_BldDataIpimbV0(schema, group, idx);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV0, Psana::Ipimb::DataV1, &Psana::Bld::BldDataIpimbV0::ipimbData> Proxy1;
    boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::DataV1> > proxy1 = boost::make_shared<Proxy1>(parent);
    evt.putProxy(proxy1, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV0, Psana::Ipimb::ConfigV1, &Psana::Bld::BldDataIpimbV0::ipimbConfig> Proxy2;
    boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::ConfigV1> > proxy2 = boost::make_shared<Proxy2>(parent);
    env.configStore().putProxy(proxy2, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV0, Psana::Lusi::IpmFexV1, &Psana::Bld::BldDataIpimbV0::ipmFexData> Proxy3;
    boost::shared_ptr<PSEvt::Proxy<Psana::Lusi::IpmFexV1> > proxy3 = boost::make_shared<Proxy3>(parent);
    evt.putProxy(proxy3, src);
    
    return;

  } else if (typeName == "Bld::BldDataIpimbV1") {

    const boost::shared_ptr<PSEvt::Proxy<Psana::Bld::BldDataIpimbV1> >& parent = Bld::make_BldDataIpimbV1(schema, group, idx);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV1, Psana::Ipimb::DataV2, &Psana::Bld::BldDataIpimbV1::ipimbData> Proxy1;
    boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::DataV2> > proxy1 = boost::make_shared<Proxy1>(parent);
    evt.putProxy(proxy1, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV1, Psana::Ipimb::ConfigV2, &Psana::Bld::BldDataIpimbV1::ipimbConfig> Proxy2;
    boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::ConfigV2> > proxy2 = boost::make_shared<Proxy2>(parent);
    env.configStore().putProxy(proxy2, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataIpimbV1, Psana::Lusi::IpmFexV1, &Psana::Bld::BldDataIpimbV1::ipmFexData> Proxy3;
    boost::shared_ptr<PSEvt::Proxy<Psana::Lusi::IpmFexV1> > proxy3 = boost::make_shared<Proxy3>(parent);
    evt.putProxy(proxy3, src);
    
    return;

  } else if (typeName == "Bld::BldDataPimV1") {

    const boost::shared_ptr<PSEvt::Proxy<Psana::Bld::BldDataPimV1> >& parent = Bld::make_BldDataPimV1(schema, group, idx);
    
    typedef SubObjectProxy<Psana::Bld::BldDataPimV1, Psana::Pulnix::TM6740ConfigV2, &Psana::Bld::BldDataPimV1::camConfig> Proxy1;
    boost::shared_ptr<PSEvt::Proxy<Psana::Pulnix::TM6740ConfigV2> > proxy1 = boost::make_shared<Proxy1>(parent);
    env.configStore().putProxy(proxy1, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataPimV1, Psana::Lusi::PimImageConfigV1, &Psana::Bld::BldDataPimV1::pimConfig> Proxy2;
    boost::shared_ptr<PSEvt::Proxy<Psana::Lusi::PimImageConfigV1> > proxy2 = boost::make_shared<Proxy2>(parent);
    env.configStore().putProxy(proxy2, src);
    
    typedef SubObjectProxy<Psana::Bld::BldDataPimV1, Psana::Camera::FrameV1, &Psana::Bld::BldDataPimV1::frame> Proxy3;
    boost::shared_ptr<PSEvt::Proxy<Psana::Camera::FrameV1> > proxy3 = boost::make_shared<Proxy3>(parent);
    evt.putProxy(proxy3, src);
    
    return;

  }

  hdfConvert(group, idx, typeName, schema, src, evt, env.configStore());
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void
HdfConverter::convertEpics(const hdf5pp::Group& group, uint64_t idx, PSEnv::EpicsStore& eStore)
{
  MsgLog(logger, debug, "HdfConverter::convertEpics - group: " << group);
  
  const std::string& gname = group.name();
  if (::isEpics(gname)) {

    // open/cache "data" dataset
    hdf5pp::DataSet ds;
    std::map<std::string, hdf5pp::DataSet>::const_iterator it = m_epicsDSCache.find(gname);
    if (it != m_epicsDSCache.end()) {
      ds = it->second;
    } else {
      ds = group.openDataSet("data");
      m_epicsDSCache.insert(std::make_pair(gname, ds));
    }

    if (::isCtrlEpics(gname)) {
    
      // CTRL epics should mean that we are only starting reading data and 
      // epics store does not have anything yet.

      
      boost::shared_ptr<Psana::Epics::EpicsPvHeader> epics = Epics::readEpics(ds, idx);
      if (epics) {
        const Pds::Src& src = this->source(group);
        eStore.store(epics, src);
      }

    } else {

      // Non-CTRL epics should mean that CTRL was already seen, get useful info from store 
      // to avoid reasding it from a file

      
      const std::string& pvname = group.basename();
      MsgLog(logger, debug, "HdfConverter::convertEpics - Non-CTRL epics: " << pvname);
      if (boost::shared_ptr<Psana::Epics::EpicsPvHeader> hdr = eStore.getPV(pvname)) {
        boost::shared_ptr<Psana::Epics::EpicsPvHeader> epics = Epics::readEpics(ds, idx, *hdr);
        if (epics) {
          const Pds::Src& src = this->source(group);
          eStore.store(epics, src);
        }
      }
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
  m_epicsDSCache.clear();
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

    version = schemaVersion(group, ::isEpics(name) ? 2 : 1);

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
  if (::isEpics(group)) {
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
  if (levels < 0) return source(group, ::isEpics(name) ? 1 : 0);

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
