#ifndef O2OTRANSLATOR_EPICSDATATYPECVT_H
#define O2OTRANSLATOR_EPICSDATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsDataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>
#include <utility>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/EvtDataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/epics/EpicsPvData.hh"
#include "hdf5pp/Group.h"
#include "O2OTranslator/CvtDataContFactoryEpics.h"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/**
 *  Converter type for EPICS XTC data
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class EpicsDataTypeCvt : public EvtDataTypeCvt<Pds::EpicsPvHeader> {
public:

  typedef Pds::EpicsPvHeader XtcType ;

  // Default constructor
  EpicsDataTypeCvt ( const std::string& topGroupName,
                     const ConfigObjectStore& configStore,
                     hsize_t chunk_size,
                     int deflate ) ;

  // Destructor
  virtual ~EpicsDataTypeCvt () ;

protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                                      const XtcType& data,
                                      size_t size,
                                      const Pds::TypeId& typeId,
                                      const O2OXtcSrc& src,
                                      const H5DataTypes::XtcClockTimeStamp& time ) ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) ;

  // get the name of the channel
  std::string pvName (const XtcType& data, const Pds::Src& src) ;

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTimeStamp> > XtcClockTimeCont ;
  typedef CvtDataContainer<CvtDataContFactoryEpics> DataCont ;

  // PV id is: (src, epics.pvId)
  typedef std::pair<Pds::Src, int> PvId;

  // compare op for PvId
  struct _PvIdCmp {
    bool operator()(const PvId& lhs, const PvId& rhs) const {
      if ( lhs.first.log() < rhs.first.log() ) return true ;
      if ( lhs.first.log() > rhs.first.log() ) return false ;
      if ( lhs.first.phy() < rhs.first.phy() ) return true ;
      if ( lhs.first.phy() > rhs.first.phy() ) return false ;
      if ( lhs.second < rhs.second ) return true ;
      return false ;
    }
  };

  struct _pvdata {
    _pvdata() : timeCont(0), dataCont(0) {}
    _pvdata(XtcClockTimeCont* tc, DataCont* dc) : timeCont(tc), dataCont(dc) {}
    XtcClockTimeCont* timeCont ;
    DataCont* dataCont ;
  };

  typedef std::map<std::string, hdf5pp::Group> PV2Group ; // maps PV name to group
  typedef std::map<std::string, hdf5pp::Type> PV2Type ;   // maps PV name to its HDF5 type
  typedef std::map<hdf5pp::Group, PV2Group> Subgroups ;   // maps Src group to (PV id -> Group) mapping
  typedef std::map<hdf5pp::Group, PV2Type> Types ;        // maps Src group to (PV id -> Type) mapping
  typedef std::map<std::string, _pvdata> PVDataMap ;      // maps PV name to containers
  typedef std::map<PvId, std::string, _PvIdCmp> PVNameMap;// maps PV id to its name

  // Data members
  const ConfigObjectStore& m_configStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  Subgroups m_subgroups ;  // maps top EPICS group (.../Epics::EpicsPv/EpicsArch.0:NoDevice.0) to PV2Group mapping
  Types m_types ;
  PVDataMap m_pvdatamap ;
  PVNameMap m_pvnames ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EPICSDATATYPECVT_H
