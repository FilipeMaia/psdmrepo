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
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Special converter class for Pds::Fli::FrameV1 XTC class
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

  typedef EvtDataTypeCvt<typename FrameType::XtcType> Super;
  typedef FrameType H5Type ;
  typedef typename FrameType::XtcType XtcType ;

  // constructor
  FliFrameV1Cvt ( const hdf5pp::Group& group,
                  const std::string& typeGroupName,
                  const Pds::Src& src,
                  const ConfigObjectStore& configStore,
                  const CvtOptions& cvtOptions,
                  int schemaVersion ) ;

  // Destructor
  virtual ~FliFrameV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  // fill containers for missing data
  virtual void fillMissing(hdf5pp::Group group,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5Type> FrameCont;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<FrameCont> m_frameCont ;
  boost::shared_ptr<FrameDataCont> m_frameDataCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_FLIFRAMEV1CVT_H
