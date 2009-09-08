#ifndef O2OTRANSLATOR_EVTDATATYPECVTDEF_H
#define O2OTRANSLATOR_EVTDATATYPECVTDEF_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtDataTypeCvtDef.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/EvtDataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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

template <typename H5Type>
class EvtDataTypeCvtDef : public EvtDataTypeCvt<typename H5Type::XtcType> {
public:

  typedef typename H5Type::XtcType XtcType ;

  // Default constructor
  EvtDataTypeCvtDef ( const std::string& typeGroupName,
                      hsize_t chunk_size,
                      int deflate )
    : EvtDataTypeCvt<typename H5Type::XtcType>( typeGroupName )
    , m_dataCont(0)
    , m_timeCont(0)
  {
    // make container for data objects
    CvtDataContFactoryDef<H5Type> dataContFactory( "data", chunk_size, deflate ) ;
    m_dataCont = new DataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", chunk_size, deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;
  }

  // Destructor
  virtual ~EvtDataTypeCvtDef () {
    delete m_dataCont ;
    delete m_timeCont ;
  }

protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                              const XtcType& data,
                              const Pds::TypeId& typeId,
                              const Pds::DetInfo& detInfo,
                              const H5DataTypes::XtcClockTime& time )
  {
    m_dataCont->container( group )->append ( H5Type(data) ) ;
    m_timeCont->container( group )->append ( time ) ;
  }

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) {
    m_dataCont->closeGroup( group ) ;
    m_timeCont->closeGroup( group ) ;
  }

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > DataCont ;

  // Data members
  DataCont* m_dataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  EvtDataTypeCvtDef ( const EvtDataTypeCvtDef& ) ;
  EvtDataTypeCvtDef operator = ( const EvtDataTypeCvtDef& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVTDEF_H
