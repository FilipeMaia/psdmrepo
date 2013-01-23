#ifndef O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
#define O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV1Cvt.
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
#include "H5DataTypes/PrincetonFrameV1.h"
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/**
 *  Special converter class for Pds::Princeton::FrameV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class PrincetonFrameV1Cvt : public EvtDataTypeCvt<Pds::Princeton::FrameV1> {
public:

  typedef Pds::Princeton::FrameV1 XtcType ;
  typedef H5DataTypes::PrincetonFrameV1 H5Type ;

  // constructor
  PrincetonFrameV1Cvt ( const hdf5pp::Group& group,
                        const std::string& typeGroupName,
                        const Pds::Src& src,
                        const ConfigObjectStore& configStore,
                        const CvtOptions& cvtOptions ) ;

  // Destructor
  virtual ~PrincetonFrameV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5Type> FrameCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  FrameCont* m_frameCont ;
  FrameDataCont* m_frameDataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
