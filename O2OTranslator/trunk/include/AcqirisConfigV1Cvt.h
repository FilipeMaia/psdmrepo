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
  AcqirisConfigV1Cvt(const hdf5pp::Group& group, const std::string& typeGroupName,
      const Pds::Src& src, const CvtOptions& cvtOptions, int schemaVersion) ;

  // Destructor
  virtual ~AcqirisConfigV1Cvt () ;

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

  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisConfigV1> ConfigCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisHorizV1> HorizCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisTrigV1> TrigCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisVertV1> VertCont ;

  // Data members
  boost::shared_ptr<ConfigCont> m_configCont ;
  boost::shared_ptr<HorizCont> m_horizCont ;
  boost::shared_ptr<TrigCont> m_trigCont ;
  boost::shared_ptr<VertCont> m_vertCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISCONFIGV1CVT_H
