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
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/epics/EpicsPvData.hh"
#include "hdf5pp/Group.h"
#include "O2OTranslator/CvtDataContFactoryEpics.h"

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

class EpicsDataTypeCvt : public DataTypeCvt<Pds::EpicsPvHeader> {
public:

  typedef Pds::EpicsPvHeader XtcType ;

  // Default constructor
  EpicsDataTypeCvt ( hdf5pp::Group group,
      const std::string& topGroupName,
      const Pds::Src& src,
      const ConfigObjectStore& configStore,
      hsize_t chunk_size,
      int deflate,
      int schemaVersion ) ;

  // Destructor
  virtual ~EpicsDataTypeCvt () ;

protected:

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage );

  // method called to fill void spaces for missing data
  virtual void missingConvert(const Pds::TypeId& typeId,
                      const O2OXtcSrc& src,
                      const H5DataTypes::XtcClockTimeStamp& time,
                      Pds::Damage damage)
  {
    // For EPICS we do not do anything if data is missing/damaged
  }

  // get the name of the channel
  std::string pvName (const XtcType& data, const Pds::Src& src) ;

  // get alias name for a Pv, return empty string if none defined
  std::string aliasName(int pvId, const Pds::Src& src);

private:

  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTimeStamp> XtcClockTimeCont ;
  typedef H5DataTypes::ObjectContainer<Pds::EpicsPvHeader> DataCont ;

  struct _pvdata {
    _pvdata() : timeCont(0), dataCont(0) {}
    _pvdata(XtcClockTimeCont* tc, DataCont* dc) : timeCont(tc), dataCont(dc) {}
    XtcClockTimeCont* timeCont ;
    DataCont* dataCont ;
  };

  typedef std::map<std::string, hdf5pp::Group> PV2Group ; // maps PV name to group
  typedef std::map<std::string, hdf5pp::Type> PV2Type ;   // maps PV name to its HDF5 type
  typedef std::map<std::string, _pvdata> PVDataMap ;      // maps PV name to containers

  // Data members
  const std::string m_typeGroupName ;
  const ConfigObjectStore& m_configStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  hdf5pp::Group m_group;
  PV2Group m_subgroups ;
  PV2Type m_types ;
  PVDataMap m_pvdatamap ;
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EPICSDATATYPECVT_H
