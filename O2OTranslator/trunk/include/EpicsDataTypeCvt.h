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
                                      const H5DataTypes::XtcClockTime& time ) ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) ;

  // generate the name for the subgroup
  std::string _subname ( const XtcType& data ) ;

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;
  typedef CvtDataContainer<CvtDataContFactoryEpics> DataCont ;

  struct _pvdata {
    _pvdata() : timeCont(0), dataCont(0) {}
    _pvdata(XtcClockTimeCont* tc, DataCont* dc) : timeCont(tc), dataCont(dc) {}
    XtcClockTimeCont* timeCont ;
    DataCont* dataCont ;
  };

  typedef std::map<int16_t,hdf5pp::Group> PV2Group ;
  typedef std::map<int16_t,hdf5pp::Type> PV2Type ;
  typedef std::map<hdf5pp::Group,PV2Group> Subgroups ;
  typedef std::map<hdf5pp::Group,PV2Type> Types ;
  typedef std::map<int16_t,_pvdata> PVDataMap ;
  typedef std::map<int16_t,std::string> PVNameMap ;
  typedef std::map<std::string,int16_t> PVName2Id ;

  // Data members
  hsize_t m_chunk_size ;
  int m_deflate ;
  Subgroups m_subgroups ;
  Types m_types ;
  PVDataMap m_pvdatamap ;
  PVNameMap m_pvnames ;
  PVName2Id m_name2id ;

  // Copy constructor and assignment are disabled by default
  EpicsDataTypeCvt ( const EpicsDataTypeCvt& ) ;
  EpicsDataTypeCvt& operator = ( const EpicsDataTypeCvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EPICSDATATYPECVT_H
