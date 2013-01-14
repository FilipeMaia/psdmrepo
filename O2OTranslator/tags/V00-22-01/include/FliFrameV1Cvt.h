#ifndef O2OTRANSLATOR_FLIFRAMEV1CVT_H
#define O2OTRANSLATOR_FLIFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FliFrameV1Cvt.
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
#include "O2OTranslator/CvtDataContFactoryTyped.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/**
 *  Special converter class for Pds::Fli::FrameV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */
template <typename FrameType>
class FliFrameV1Cvt : public EvtDataTypeCvt<typename FrameType::XtcType> {
public:

  typedef FrameType H5Type ;
  typedef typename FrameType::XtcType XtcType ;

  // constructor
  FliFrameV1Cvt ( const std::string& typeGroupName,
                  const ConfigObjectStore& configStore,
                  Pds::TypeId cfgTypeId,
                  hsize_t chunk_size,
                  int deflate ) ;

  // Destructor
  virtual ~FliFrameV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > FrameCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  Pds::TypeId m_cfgTypeId;
  FrameCont* m_frameCont ;
  FrameDataCont* m_frameDataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_FLIFRAMEV1CVT_H
