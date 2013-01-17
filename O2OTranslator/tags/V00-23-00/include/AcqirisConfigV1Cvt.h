#ifndef O2OTRANSLATOR_ACQIRISCONFIGV1CVT_H
#define O2OTRANSLATOR_ACQIRISCONFIGV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisConfigV1Cvt.
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
#include "H5DataTypes/AcqirisConfigV1.h"
#include "pdsdata/acqiris/ConfigV1.hh"
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

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Special coverter class for Acqiris confugration class.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class AcqirisConfigV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::ConfigV1> {
public:

  typedef Pds::Acqiris::ConfigV1 XtcType ;

  // Default constructor
  AcqirisConfigV1Cvt ( const std::string& typeGroupName,
                       hsize_t chunk_size,
                       int deflate,
                       SrcFilter srcFilter = SrcFilter::allow(SrcFilter::BLD) ) ;

  // Destructor
  virtual ~AcqirisConfigV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::AcqirisConfigV1> > ConfigCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::AcqirisHorizV1> > HorizCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::AcqirisTrigV1> > TrigCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<H5DataTypes::AcqirisVertV1> > VertCont ;

  // Data members
  ConfigCont* m_configCont ;
  HorizCont* m_horizCont ;
  TrigCont* m_trigCont ;
  VertCont* m_vertCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISCONFIGV1CVT_H
