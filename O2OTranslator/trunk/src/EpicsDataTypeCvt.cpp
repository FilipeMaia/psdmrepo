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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "EpicsDataTypeCvt" ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
EpicsDataTypeCvt::EpicsDataTypeCvt ( const std::string& topGroupName,
                                     hsize_t chunk_size,
                                     int deflate )
  : EvtDataTypeCvt<Pds::EpicsPvHeader>( topGroupName )
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_subgroups()
  , m_types()
  , m_pvdatamap()
  , m_pvnames()
  , m_name2id()
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
  
  // see if there is a structure setup already for this PV
  PVDataMap::iterator pv_it = m_pvdatamap.find(data.iPvId) ;
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
    pv_it = m_pvdatamap.insert( PVDataMap::value_type(data.iPvId, pv) ).first ;
  }

  // is there a subgroup for this PV?
  hdf5pp::Group subgroup = m_subgroups[group][data.iPvId] ;
  if ( not subgroup.valid() ) {
    const std::string& subname = _subname( data ) ;
    MsgLog(logger,trace, "EpicsDataTypeCvt -- creating subgroup " << subname ) ;
    if (group.hasChild(subname)) {
      subgroup = group.openGroup( subname ) ;
    } else {
      subgroup = group.createGroup( subname ) ;
    }
    m_subgroups[group][data.iPvId] = subgroup ;
  }

  // is there a type for this PV?
  hdf5pp::Type type = m_types[group][data.iPvId] ;
  if ( not type.valid() ) {
    type = CvtDataContFactoryEpics::stored_type( data ) ;
    m_types[group][data.iPvId] = type ;
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

// generate the name for the subgroup
std::string
EpicsDataTypeCvt::_subname ( const XtcType& data )
{
  // generate new name, if the data is CTRL data then it already contains
  // EPICS name of PV, otherwise generate some unique name
  std::string name = m_pvnames[data.iPvId] ;
  if ( name.empty() ) {
    if ( dbr_type_is_CTRL(data.iDbrType) ) {

      name = static_cast<const Pds::EpicsPvCtrlHeader&>(data).sPvName ;

      // some channels may appear twice
      if ( m_name2id.count(name) > 0 ) {
        MsgLog(logger,warning,"EpicsDataTypeCvt -- duplicate PV name: " << name << " for IDs " << m_name2id[name] << " and " << data.iPvId ) ;

        // augment it with channel ID
        char buf[64] ;
        snprintf( buf, sizeof buf, "%s#%d", static_cast<const Pds::EpicsPvCtrlHeader&>(data).sPvName, int(data.iPvId) ) ;
        name = buf ;
      }

      m_pvnames[data.iPvId] = name ;
      m_name2id[name] = data.iPvId ;

    } else {

      char buf[32] ;
      snprintf( buf, sizeof buf, "pvId=%d", int(data.iPvId) ) ;
      name = buf ;

    }
  }

  return name ;
}


} // namespace O2OTranslator
