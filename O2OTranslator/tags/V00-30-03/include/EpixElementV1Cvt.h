#ifndef O2OTRANSLATOR_EPIXELEMENTV1CVT_H
#define O2OTRANSLATOR_EPIXELEMENTV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixElementV1Cvt.
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
#include "H5DataTypes/EpixElementV1.h"
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
 *  @brief Special converter class for Pds::Epix::ElementV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class EpixElementV1Cvt : public EvtDataTypeCvt<Pds::Epix::ElementV1> {
public:

  typedef Pds::Epix::ElementV1 XtcType;
  typedef H5DataTypes::EpixElementV1 H5Type;
  
  // constructor
  EpixElementV1Cvt(const hdf5pp::Group& group,
      const std::string& typeGroupName,
      const Pds::Src& src,
      const ConfigObjectStore& configStore,
      const CvtOptions& cvtOptions,
      int schemaVersion);

  // Destructor
  virtual ~EpixElementV1Cvt () ;

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

  // finds config object and gets various dimensions from it
  std::vector<int> shape(const O2OXtcSrc& src);

  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> ExcludedRowsCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> TemperatureCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<FrameCont> m_frameCont ;
  boost::shared_ptr<ExcludedRowsCont> m_excludedRowsCont ;
  boost::shared_ptr<TemperatureCont> m_temperatureCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EPIXELEMENTV1CVT_H
