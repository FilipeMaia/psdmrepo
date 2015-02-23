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
#include "psddl_hdf2psana/SchemaConstants.h"
//-----------------
// C/C++ Headers --
//-----------------
#include <stdlib.h>
#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/NameIter.h"
#include "hdf5pp/Utils.h"
#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "psddl_hdf2psana/HdfGroupName.h"
#include "psddl_hdf2psana/Exceptions.h"
#include "psddl_hdf2psana/dispatch.h"
#include "psddl_hdf2psana/bld.ddl.h"
#include "psddl_hdf2psana/epics.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psddl_hdf2psana;

namespace {

  const std::string logger = "psddl_hdf2psana.HdfConverter";

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
  
  void setNDArrayParamsFromGroupAttrs(const hdf5pp::Group &group, 
                                      NDArrayParameters &ndarrayParams) {

    // dimension
    hdf5pp::Attribute<uint32_t> dimAttr = group.openAttr<uint32_t>(ndarrayDimAttrName);
    if (dimAttr.valid()) {
      ndarrayParams.dim(dimAttr.read());
    } else {
      MsgLog(logger, error, " setNDArrayParamsFromGroupAttrs: " << ndarrayDimAttrName 
             << " is not valid in " << group.name());
    }

    // element type
    hdf5pp::Attribute<int16_t> elemTypeAttr = group.openAttr<int16_t>(ndarrayElemTypeAttrName);
    if (elemTypeAttr.valid()) {
      ndarrayParams.elemType(NDArrayParameters::ElemType(elemTypeAttr.read()));
    } else {
      MsgLog(logger, error, " setNDArrayParamsFromGroupAttrs: " << ndarrayElemTypeAttrName 
             << " is not valid in " << group.name());
    }

    // element size bytes
    hdf5pp::Attribute<uint32_t> sizeBytesAttr = group.openAttr<uint32_t>(ndarraySizeBytesAttrName);
    if (sizeBytesAttr.valid()) {
      ndarrayParams.sizeBytes(sizeBytesAttr.read());
    } else {
      MsgLog(logger, error, " setNDArrayParamsFromGroupAttrs: " << ndarraySizeBytesAttrName 
             << " is not valid in " << group.name());
    }

    // const elem
    hdf5pp::Attribute<uint8_t> isConstAttr = group.openAttr<uint8_t>(ndarrayConstElemAttrName);
    if (isConstAttr.valid()) {
      ndarrayParams.isConstElem(isConstAttr.read());
    } else {
      MsgLog(logger, error, " setNDArrayParamsFromGroupAttrs: " << ndarrayConstElemAttrName 
             << " is not valid in " << group.name());
    }

    // vlen slow dim
    hdf5pp::Attribute<uint8_t> vlenAttr = group.openAttr<uint8_t>(vlenAttrName);
    if (vlenAttr.valid()) {
      ndarrayParams.isVlen(vlenAttr.read());
    } else {
      MsgLog(logger, error, " setNDArrayParamsFromGroupAttrs: " << vlenAttrName  
             << " is not valid in " << group.name());
    }
    MsgLog(logger, debug, "group=" << group.name() << " ndarrayParams: "
           << " elemType=" << ndarrayParams.elemType()
           << " sizeBytes=" << ndarrayParams.sizeBytes()
           << " dim=" << ndarrayParams.dim()
           << " const=" << ndarrayParams.isConstElem()
           << " vlen=" << ndarrayParams.isVlen());
  }

  std::string readH5stringAttribute(const std::string &attrName, const hid_t groupId, bool &success) {
    success = false;
    std::string toReturn;
    htri_t rc = H5Aexists(groupId, attrName.c_str());
    if ( rc < 0 ) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Aexists"); 
    if ( rc > 0) {
      hid_t attrId = H5Aopen (groupId, attrName.c_str(), H5P_DEFAULT);
      if ( attrId < 0 ) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Aopen");
      hid_t fileType = H5Aget_type(attrId);
      H5T_class_t type_class = H5Tget_class(fileType);
      if (type_class == H5T_STRING) {
        if (true == H5Tis_variable_str(fileType)) {
          hid_t memType = H5Tget_native_type(fileType, H5T_DIR_ASCEND);
          if (memType < 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"H5Tget_native_type");
          hid_t attrDataSpace = H5Aget_space(attrId);
          if (attrDataSpace < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Aget_space");
          hssize_t size = H5Sget_simple_extent_npoints(attrDataSpace);
          if (size == 1) {
            char * stringData=NULL;
            if (H5Aread(attrId, memType, &stringData) < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Aread");
            toReturn = stringData;
            free(stringData);
          } else {
            MsgLog(logger, warning, "readH5stringAttribute: dataspace size != 1, it is " << size);
          }
          if (H5Sclose(attrDataSpace)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Sclose");
          if (H5Tclose(memType)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose");
        } else {
          MsgLog(logger, warning, "readH5stringAttribute: non-variable length string. currently not supported");
        }
        if (H5Tclose(fileType)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Tclose");
      }
      if (H5Aclose(attrId)<0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Aclose");
    }
    return toReturn;
  }

  std::string getEventKey(const hdf5pp::Group & group) {
    bool success=false;
    std::string eventKeyStr = readH5stringAttribute(h5GroupNameKeyAttrName, group.id(), success);
    if (not success) {
      // directly parse from group name
      std::string key = group.name();
      std::string::size_type p = key.rfind('/');
      key.erase(0,p);
      p = key.find(srcEventKeySeperator);
      if (p == std::string::npos) return "";
      key.erase(0,p+srcEventKeySeperator.size());
      eventKeyStr = key;
    }
    return eventKeyStr;
  }

} // local namespace

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
HdfConverter::convert(const hdf5pp::Group& group, int64_t idx, PSEvt::Event& evt, PSEnv::Env& env)
{
  if (::isEpics(group.name())) return;

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

    // Special case for NDArray
  } else if (isNDArray(group)) {
    NDArrayParameters ndarrayParams;
    setNDArrayParamsFromGroupAttrs(group.parent(), ndarrayParams);
    std::string key = getEventKey(group);
    m_ndarrayConverter.convert(group, idx, ndarrayParams, schema, src, key, evt);
    return;
  }

  hdfConvert(group, idx, typeName, schema, src, evt, env.configStore());
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void
HdfConverter::convertEpics(const hdf5pp::Group& group, int64_t idx, PSEnv::EpicsStore& eStore)
{
  MsgLog(logger, debug, "HdfConverter::convertEpics - group: " << group << " index: " << idx);
  
  const std::string& gname = group.name();
  if (::isEpics(gname)) {

    if (idx < 0) {

      // Special case, this is the group which contain name of all epics PVs
      // and aliases, we need to extract aliases and add them to the epics store.
      // Aliases are sof links pointing to the original PV names

      // also need source which is determined by the name of this group
      Pds::Src src = this->source(group, 0);

      hdf5pp::NameIter names(group, hdf5pp::NameIter::SoftLink);
      for (std::string alias = names.next(); not alias.empty(); alias = names.next()) {

        const std::string& pvName = group.getSoftLink(alias);
        if (not pvName.empty()) {

          // we need PV id which we have to read from a dataset
          hdf5pp::Group pvGroup = group.openGroup(pvName);
          hdf5pp::DataSet ds = pvGroup.openDataSet("data");
          boost::shared_ptr<Psana::Epics::EpicsPvHeader> epics = Epics::readEpics(ds, 0);
          if (epics) {
            eStore.storeAlias(src, epics->pvId(), alias);
          }
        }
      }

    } else {

      // open "data" dataset
      hdf5pp::DataSet ds = group.openDataSet("data");

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
        // to avoid reading it from a file


        const std::string& pvname = group.basename();
        MsgLog(logger, debug, "HdfConverter::convertEpics - Non-CTRL epics: " << pvname);
        if (boost::shared_ptr<Psana::Epics::EpicsPvHeader> hdr = eStore.getPV(pvname)) {
          boost::shared_ptr<Psana::Epics::EpicsPvHeader> epics = Epics::readEpics(ds, idx, *hdr);
          if (epics) {
            const Pds::Src& src = this->source(group);
            eStore.store(epics, src, &pvname);
          }
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
  hdf5pp::Attribute<int> attr = group.openAttr<int>(versionAttrName);
  if (attr.valid()) {
    version = attr.read();
    MsgLog(logger, debug, "got schema version " << version << " from group attribute: " << name);
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) version = schemaVersion(parent, levels - 1);
  } else {
    MsgLog(logger, debug, "use default schema version 0 for group: " << name);
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
  hdf5pp::Attribute<uint64_t> attrSrc = group.openAttr<uint64_t>(srcAttrName);
  if (attrSrc.valid()) {
    // build source from attributes
    src = ::_SrcBuilder(attrSrc.read());
    MsgLog(logger, debug, "got source " << src << " from group attribute: " << name);
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) src = this->source(parent, levels - 1);
  } else if (levels == 0) {
    // guess source from group name for top-level type group
    src = HdfGroupName::nameToSource(group.basename());

    // some corrections needed for incorrectly stored names
    if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 0)) {
      src = Pds::BldInfo(0, Pds::BldInfo::EBeam);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 1)) {
      src = Pds::BldInfo(0, Pds::BldInfo::PhaseCavity);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 2)) {
      src = Pds::BldInfo(0, Pds::BldInfo::FEEGasDetEnergy);
    }
    MsgLog(logger, debug, "got source " << src << " from group name: " << name);
  }

  // update cache
  m_sourceCache.insert(std::make_pair(name, src));

  return src;
}

bool HdfConverter::isNDArray(const hdf5pp::Group& group) const
{
  hdf5pp::Group parent = group.parent();
  hdf5pp::Attribute<uint8_t> attr = parent.openAttr<uint8_t>(ndarrayAttrName);
  if (attr.valid()) {
    uint8_t isNDArray = attr.read();
    return (isNDArray >= 1);
  } 
  return false;
}

} // namespace psddl_hdf2psana
