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
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  Special converter class for Pds::CsPad::DataV2 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadElementV2Cvt : public EvtDataTypeCvt<Pds::CsPad::DataV2> {
public:

  typedef Pds::CsPad::DataV2 XtcType ;
  typedef H5DataTypes::CsPadElementV2 H5Type ;

  // constructor
  CsPadElementV2Cvt ( const hdf5pp::Group& group,
                      const std::string& typeGroupName,
                      const Pds::Src& src,
                      const ConfigObjectStore& configStore,
                      const CalibObjectStore& calibStore,
                      const CvtOptions& cvtOptions,
                      int schemaVersion ) ;

  // Destructor
  virtual ~CsPadElementV2Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  // typed conversion method templated on Configuration type
  template <typename Config>
  void fillContainers(hdf5pp::Group group,
                      const XtcType& data,
                      size_t size,
                      const Pds::TypeId& typeId,
                      const O2OXtcSrc& src,
                      const Config& cfg);

  // fill containers for missing data
  virtual void fillMissing(hdf5pp::Group group,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5Type> ElementCont ;
  typedef H5DataTypes::ObjectContainer<int16_t> PixelDataCont ;
  typedef H5DataTypes::ObjectContainer<float> CommonModeDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  const CalibObjectStore& m_calibStore;
  boost::shared_ptr<ElementCont> m_elementCont ;
  boost::shared_ptr<PixelDataCont> m_pixelDataCont ;
  boost::shared_ptr<CommonModeDataCont> m_cmodeDataCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPADELEMENTV2CVT_H
