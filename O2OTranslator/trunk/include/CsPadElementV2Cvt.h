#ifndef O2OTRANSLATOR_CSPADELEMENTV2CVT_H
#define O2OTRANSLATOR_CSPADELEMENTV2CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2Cvt.
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
#include "H5DataTypes/CsPadElementV2.h"
#include "O2OTranslator/CalibObjectStore.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"
#include "O2OTranslator/CvtDataContFactoryTyped.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::CsPad::ElementV2 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadElementV2Cvt : public EvtDataTypeCvt<Pds::CsPad::ElementV2> {
public:

  typedef Pds::CsPad::ElementV2 XtcType ;

  // constructor
  CsPadElementV2Cvt ( const std::string& typeGroupName,
                      const ConfigObjectStore& configStore,
                      const CalibObjectStore& calibStore,
                      hsize_t chunk_size,
                      int deflate ) ;

  // Destructor
  virtual ~CsPadElementV2Cvt () ;

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

private:

  typedef CvtDataContainer<CvtDataContFactoryTyped<H5DataTypes::CsPadElementV2> > ElementCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<int16_t> > PixelDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<float> > CommonModeDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  const CalibObjectStore& m_calibStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  ElementCont* m_elementCont ;
  PixelDataCont* m_pixelDataCont ;
  CommonModeDataCont* m_cmodeDataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  CsPadElementV2Cvt ( const CsPadElementV2Cvt& ) ;
  CsPadElementV2Cvt& operator = ( const CsPadElementV2Cvt& ) ;
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPADELEMENTV2CVT_H
