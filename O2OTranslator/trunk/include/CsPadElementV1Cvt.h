#ifndef O2OTRANSLATOR_CSPADELEMENTV1CVT_H
#define O2OTRANSLATOR_CSPADELEMENTV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1Cvt.
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
#include "H5DataTypes/CsPadElementV1.h"
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
 *  Special converter class for Pds::CsPad::ElementV1 XTC class
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

class CsPadElementV1Cvt : public EvtDataTypeCvt<Pds::CsPad::ElementV1> {
public:

  typedef Pds::CsPad::ElementV1 XtcType ;
  typedef H5DataTypes::CsPadElementV1 H5Type ;

  // constructor
  CsPadElementV1Cvt ( const std::string& typeGroupName,
                      const ConfigObjectStore& configStore,
                      const CalibObjectStore& calibStore,
                      hsize_t chunk_size,
                      int deflate ) ;

  // Destructor
  virtual ~CsPadElementV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hsize_t chunk_size, int deflate,
      const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  /// method called when the driver closes a group in the file
  virtual void closeContainers(hdf5pp::Group group);

private:

  typedef CvtDataContainer<CvtDataContFactoryTyped<H5Type> > ElementCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<int16_t> > PixelDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<float> > CommonModeDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  const CalibObjectStore& m_calibStore;
  ElementCont* m_elementCont ;
  PixelDataCont* m_pixelDataCont ;
  CommonModeDataCont* m_cmodeDataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPADELEMENTV1CVT_H
