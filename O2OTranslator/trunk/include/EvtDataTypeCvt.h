#ifndef O2OTRANSLATOR_EVTDATATYPECVT_H
#define O2OTRANSLATOR_EVTDATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtDataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/ObjectContainer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Data converter class for event-type data objects
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
class EvtDataTypeCvt : public DataTypeCvt<typename H5Type::XtcType> {
public:

  typedef typename H5Type::XtcType XtcType ;

  // constructor takes a location where the data will be stored
  EvtDataTypeCvt ( hdf5pp::Group group,
                   hsize_t chunk_size,
                   int deflate )
    : DataTypeCvt<typename H5Type::XtcType>()
    , m_dataCont(0)
    , m_timeCont(0)
  {
    // make container for data objects
    hsize_t chunk = std::max ( chunk_size / sizeof (H5Type), hsize_t(1) ) ;
    MsgLog( "EvtDataTypeCvt", debug, "chunk size for data: " << chunk ) ;
    m_dataCont = new DataCont ( "data", group, chunk, deflate ) ;

    // make container for time
    chunk = std::max ( chunk_size / sizeof(H5DataTypes::XtcClockTime), hsize_t(1) ) ;
    MsgLog( "EvtDataTypeCvt", debug, "chunk size for time: " << chunk ) ;
    m_timeCont = new XtcClockTimeCont ( "time", group, chunk, deflate ) ;
  }

  // Destructor
  virtual ~EvtDataTypeCvt ()
  {
    delete m_dataCont ;
    delete m_timeCont ;
  }

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              const H5DataTypes::XtcClockTime& time )
  {
    // store the data in the containers
    m_dataCont->append ( H5Type(data) ) ;
    m_timeCont->append ( time ) ;
  }

protected:

private:

  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTime> XtcClockTimeCont ;
  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;

  // Data members
  DataCont* m_dataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  EvtDataTypeCvt ( const EvtDataTypeCvt& ) ;
  EvtDataTypeCvt operator = ( const EvtDataTypeCvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVT_H
