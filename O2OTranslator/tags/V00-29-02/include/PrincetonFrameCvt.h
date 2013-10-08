#ifndef O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
#define O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameCvt.
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
 *  Special converter class for Princeton frame XTC class,
 *  this converter works for both FrameV1 and FrameV2 classes.
 *  Its template argument should be either H5DataTypes::PrincetonFrameV1
 *  or H5DataTypes::PrincetonFrameV2
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */
template <typename H5DataType>
class PrincetonFrameCvt : public EvtDataTypeCvt<typename H5DataType::XtcType> {
public:

  typedef typename H5DataType::XtcType XtcType ;
  typedef H5DataType H5Type ;

  // constructor
  PrincetonFrameCvt ( const hdf5pp::Group& group,
                        const std::string& typeGroupName,
                        const Pds::Src& src,
                        const ConfigObjectStore& configStore,
                        const CvtOptions& cvtOptions,
                        int schemaVersion ) ;

  // Destructor
  virtual ~PrincetonFrameCvt () ;

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

  typedef H5DataTypes::ObjectContainer<H5Type> FrameCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<FrameCont> m_frameCont ;
  boost::shared_ptr<FrameDataCont> m_frameDataCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
