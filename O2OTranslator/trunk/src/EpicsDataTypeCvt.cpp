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

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EpicsDataTypeCvt::EpicsDataTypeCvt ( const std::string& topGroupName,
                                     const ConfigObjectStore& configStore,
                                     hsize_t chunk_size,
                                     int deflate )
  : EvtDataTypeCvt<Pds::EpicsPvHeader>( topGroupName )
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_subgroups()
  , m_types()
  , m_pvdatamap()
  , m_pvnames()
{
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
EpicsDataTypeCvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  MsgLog(logger,debug, "EpicsDataTypeCvt -- pv id = " << data.iPvId ) ;

  if (size == 0) {
    // Rare form of data damage
    MsgLog("ConfigDataTypeCvt", warning, "Zero XTC payload in " << typeGroupName()) ;
    return;
  }

  // get the name
  const std::string pvname = pvName(data, src.top());

  // see if there is a structure setup already for this PV
  PVDataMap::iterator pv_it = m_pvdatamap.find(pvname) ;
  if ( pv_it == m_pvdatamap.end() ) {
    // new thing, create all groups/containers

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
    XtcClockTimeCont* timeCont = new XtcClockTimeCont ( timeContFactory ) ;

    // make container for data objects
    CvtDataContFactoryEpics dataContFactory( "data", m_chunk_size, m_deflate, false ) ;
    DataCont* dataCont = new DataCont ( dataContFactory ) ;

    // store it all for later use
    _pvdata pv( timeCont, dataCont ) ;
    pv_it = m_pvdatamap.insert( std::make_pair(pvname, pv) ).first ;
  }

  // is there a subgroup for this PV?
  hdf5pp::Group subgroup = m_subgroups[group][pvname] ;
  if ( not subgroup.valid() ) {
    if (group.hasChild(pvname)) {

      MsgLog(logger,trace, "EpicsDataTypeCvt -- opening subgroup " << pvname ) ;
      subgroup = group.openGroup( pvname ) ;

    } else {

      MsgLog(logger,trace, "EpicsDataTypeCvt -- creating subgroup " << pvname ) ;
      subgroup = group.createGroup( pvname ) ;

      // there may be an alias defined for this PV
      Pds::TypeId typeId(Pds::TypeId::Id_EpicsConfig, 1);
      if (const Pds::Epics::ConfigV1* ecfg = m_configStore.find<Pds::Epics::ConfigV1>(typeId, src.top())) {
        for (int i = 0; i != ecfg->getNumPv(); ++ i) {
          const Pds::Epics::PvConfigV1& pvcfg = *ecfg->getPvConfig(i);
          if (pvcfg.iPvId == data.iPvId) {
            const std::string alias = ::normAliasName(pvcfg.sPvDesc);
            if (pvname == alias) {
              MsgLog(logger, debug, "EpicsDataTypeCvt -- alias is the same as PV name " << pvcfg.sPvDesc );
            } else if (group.hasChild(alias)) {
              MsgLog(logger, warning, "EpicsDataTypeCvt -- alias has the same name as another PV or alias name: " << alias );
            } else {
              try {
                MsgLog(logger,trace, "EpicsDataTypeCvt -- creating alias " << alias ) ;
                group.makeSoftLink(pvname, alias);
              } catch (const hdf5pp::Exception& ex) {
                // complain but continue
                MsgLog(logger,trace, "EpicsDataTypeCvt -- failed to create alias \"" << alias << "\": " << ex.what()) ;
              }
              break;
            }
          }
        }
      }

    }
    m_subgroups[group][pvname] = subgroup ;
  }

  // is there a type for this PV?
  hdf5pp::Type type = m_types[group][pvname] ;
  if ( not type.valid() ) {
    type = CvtDataContFactoryEpics::stored_type( data ) ;
    m_types[group][pvname] = type ;
  }

  _pvdata& pv = pv_it->second ;
  pv.timeCont->container(subgroup)->append(time) ;
  pv.dataCont->container(subgroup,data)->append(data,type) ;
}

/// method called when the driver closes a group in the file
void
EpicsDataTypeCvt::closeSubgroup( hdf5pp::Group group )
{
  // close all subgroups
  PV2Group& pv2group = m_subgroups[group] ;
  for ( PV2Group::const_iterator it = pv2group.begin() ; it != pv2group.end() ; ++ it ) {
    PVDataMap::iterator dit = m_pvdatamap.find(it->first) ;
    if ( dit != m_pvdatamap.end() ) {
      dit->second.timeCont->closeGroup( it->second ) ;
      dit->second.dataCont->closeGroup( it->second ) ;
    }
  }

  // forget about this group
  m_subgroups.erase( group ) ;
  m_types.erase( group ) ;
}

// get the name of the channel
std::string
EpicsDataTypeCvt::pvName (const XtcType& data, const Pds::Src& src)
{
  PvId pvid(src, data.iPvId);

  std::string name = m_pvnames[pvid] ;
  if ( name.empty() ) {

    if ( dbr_type_is_CTRL(data.iDbrType) ) {

      name = static_cast<const Pds::EpicsPvCtrlHeader&>(data).sPvName ;

    } else {

      name = "PV:pvId=" + boost::lexical_cast<std::string>(data.iPvId) +
          ":src_log=" + boost::lexical_cast<std::string>(src.log()) +
          ":src_phy=" + boost::lexical_cast<std::string>(src.phy());

    }

    m_pvnames[pvid] = name ;
  }

  return name ;
}


} // namespace O2OTranslator
