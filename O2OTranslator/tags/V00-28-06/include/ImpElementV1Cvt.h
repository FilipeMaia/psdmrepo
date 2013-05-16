#ifndef O2OTRANSLATOR_IMPELEMENTV1CVT_H
#define O2OTRANSLATOR_IMPELEMENTV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpElementV1Cvt.
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
#include "H5DataTypes/ImpElementV1.h"
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
 *  Special converter class for Pds::Imp::ElementV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ImpElementV1Cvt : public EvtDataTypeCvt<Pds::Imp::ElementV1> {
public:

  typedef Pds::Imp::ElementV1 XtcType;
  typedef H5DataTypes::ImpElementV1 H5Type;
  typedef H5DataTypes::ImpSample H5SampleType;
  
  // constructor
  ImpElementV1Cvt(const hdf5pp::Group& group,
      const std::string& typeGroupName,
      const Pds::Src& src,
      const ConfigObjectStore& configStore,
      const CvtOptions& cvtOptions,
      int schemaVersion);

  // Destructor
  virtual ~ImpElementV1Cvt () ;

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

  // finds config object and gets number of samples from it, returns -1 if config object is not there
  int nSamples(const O2OXtcSrc& src);
  
  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;
  typedef H5DataTypes::ObjectContainer<H5SampleType> SamplesCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<SamplesCont> m_samplesCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_IMPELEMENTV1CVT_H
