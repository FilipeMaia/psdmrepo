//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsDataTypeCvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/EpicsDataTypeCvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "hdf5pp/Exceptions.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "pdsdata/epics/ConfigV1.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "EpicsDataTypeCvt" ;

  // normalize alias name, remove special characters
  std::string normAliasName(std::string alias) {
    std::replace(alias.begin(), alias.end(), '/', '_');
    return alias;
  }

  // PV id is: (src, epics.pvId)
  typedef std::pair<Pds::Src, int> PvId;

  // compare op for PvId
  struct _PvIdCmp {
    bool operator()(const PvId& lhs, const PvId& rhs) const {
      if ( lhs.second < rhs.second ) return true ;
      if ( lhs.second > rhs.second ) return false ;
      if ( lhs.first.log() < rhs.first.log() ) return true ;
      if ( lhs.first.log() > rhs.first.log() ) return false ;
      if ( lhs.first.phy() < rhs.first.phy() ) return true ;
      return false ;
    }
  };

  typedef std::map<PvId, std::string, _PvIdCmp> PVNameMap;// maps PV id to its name

  // shared mapping from PvId to PV name
  PVNameMap g_pvnames;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EpicsDataTypeCvt::EpicsDataTypeCvt ( hdf5pp::Group group,
    const std::string& topGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    hsize_t chunk_size,
    int deflate,
    int schemaVersion)
  : DataTypeCvt<Pds::EpicsPvHeader>()
  , m_typeGroupName(topGroupName)
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_group()
  , m_subgroups()
  , m_types()
  , m_pvdatamap()
{
  // get the name of the group for this object
  const std::string& srcName = boost::lexical_cast<std::string>(src);
  const std::string& grpName = m_typeGroupName + "/" + srcName;

  // create separate group
  if (group.hasChild(grpName)) {
    MsgLog(logger, trace, "existing group " << grpName ) ;
    m_group = group.openGroup( grpName );
  } else {
    MsgLog(logger, trace, "creating group " << grpName ) ;
    m_group = group.createGroup( grpName );

    // store some group attributes
    uint64_t srcVal = (uint64_t(src.phy()) << 32) + src.log();
    m_group.createAttr<uint64_t>("_xtcSrc").store(srcVal);
    m_group.createAttr<uint32_t>("_schemaVersion").store(schemaVersion);
  }
}

//--------------
// Destructor --
//--------------
EpicsDataTypeCvt::~EpicsDataTypeCvt ()
{
  // clear allocated stuff
  for ( PVDataMap::iterator it = m_pvdatamap.begin() ; it != m_pvdatamap.end() ; ++ it ) {
    delete it->second.timeCont ;
    delete it->second.dataCont ;
  }
}

// typed conversion method
void
EpicsDataTypeCvt::typedConvert ( const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src,
    const H5DataTypes::XtcClockTimeStamp& time,
    Pds::Damage damage )
{
  MsgLog(logger,debug, "EpicsDataTypeCvt -- pv id = " << data.iPvId ) ;

  if (size == 0) {
    // Rare form of data damage
    MsgLog("ConfigDataTypeCvt", warning, "Zero XTC payload in " << m_typeGroupName) ;
    return;
  }

  // get the name
  const std::string pvname = pvName(data, src.top());

  // is there a subgroup for this PV?
  hdf5pp::Group subgroup = m_subgroups[pvname] ;
  if ( not subgroup.valid() ) {

    if (m_group.hasChild(pvname)) {

      MsgLog(logger,trace, "EpicsDataTypeCvt -- opening subgroup " << pvname ) ;
      subgroup = m_group.openGroup( pvname ) ;

    } else {

      MsgLog(logger,trace, "EpicsDataTypeCvt -- creating subgroup " << pvname ) ;
      subgroup = m_group.createGroup( pvname ) ;

    }

    m_subgroups[pvname] = subgroup ;
  }

  // there may be an alias defined for this PV
  const std::string& alias = aliasName(data.iPvId, src.top());
  if (alias.empty() or pvname == alias) {
    // fine, means nothing to do
  } else  if (m_subgroups.count(alias) == 0) {

    if (m_group.hasChild(alias)) {
      MsgLog(logger, warning, "EpicsDataTypeCvt -- alias has the same name as another PV or alias name: " << alias );
    } else {
      try {
        MsgLog(logger,trace, "EpicsDataTypeCvt -- creating alias " << alias ) ;
        m_group.makeSoftLink(pvname, alias);
      } catch (const hdf5pp::Exception& ex) {
        // complain but continue
        MsgLog(logger,trace, "EpicsDataTypeCvt -- failed to create alias \"" << alias << "\": " << ex.what()) ;
      }
    }

    m_subgroups[alias] = subgroup ;
  }

  // is there a type for this PV?
  hdf5pp::Type type = m_types[pvname] ;
  if (not type.valid()) {
    type = CvtDataContFactoryEpics::stored_type(data);
    m_types[pvname] = type ;
  }

  // see if there is a structure setup already for this PV
  PVDataMap::iterator pv_it = m_pvdatamap.find(pvname) ;
  if ( pv_it == m_pvdatamap.end() ) {
    // new thing, create all groups/containers

    // make container for time
    XtcClockTimeCont* timeCont = new XtcClockTimeCont("time", subgroup,
        XtcClockTimeCont::value_type::stored_type(), m_chunk_size, m_deflate, true);

    // make container for data objects
    DataCont* dataCont = new DataCont("data", subgroup, type, m_chunk_size, m_deflate, false);

    // store it all for later use
    _pvdata pv( timeCont, dataCont ) ;
    pv_it = m_pvdatamap.insert( std::make_pair(pvname, pv) ).first ;
  }

  _pvdata& pv = pv_it->second;
  pv.timeCont->append(time);
  pv.dataCont->append(data, type);
}

// get the name of the channel
std::string
EpicsDataTypeCvt::pvName (const XtcType& data, const Pds::Src& src)
{
  PvId pvid(src, data.iPvId);

  std::string name = g_pvnames[pvid] ;
  if ( name.empty() ) {

    if ( dbr_type_is_CTRL(data.iDbrType) ) {

      name = static_cast<const Pds::EpicsPvCtrlHeader&>(data).sPvName ;

    } else {

      name = "PV:pvId=" + boost::lexical_cast<std::string>(data.iPvId) +
          ":src_log=" + boost::lexical_cast<std::string>(src.log()) +
          ":src_phy=" + boost::lexical_cast<std::string>(src.phy());

    }

    g_pvnames[pvid] = name ;
  }

  return name ;
}

// get alias name for a Pv, return empty string if none defined
std::string
EpicsDataTypeCvt::aliasName(int pvId, const Pds::Src& src)
{
  std::string alias;

  Pds::TypeId typeId(Pds::TypeId::Id_EpicsConfig, 1);
  if (const Pds::Epics::ConfigV1* ecfg = m_configStore.find<Pds::Epics::ConfigV1>(typeId, src)) {
    for (int i = 0; i != ecfg->getNumPv(); ++ i) {
      const Pds::Epics::PvConfigV1& pvcfg = *ecfg->getPvConfig(i);
      if (pvcfg.iPvId == pvId) {
        alias = ::normAliasName(pvcfg.sPvDesc);
        break;
      }
    }
  }
  return alias;
}

} // namespace O2OTranslator
