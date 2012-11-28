#ifndef O2OTRANSLATOR_EVTDATATYPECVTDEF_H
#define O2OTRANSLATOR_EVTDATATYPECVTDEF_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtDataTypeCvtDef.
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
#include "O2OTranslator/O2OExceptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

template <typename H5Type>
class EvtDataTypeCvtDef : public EvtDataTypeCvt<typename H5Type::XtcType> {
public:

  typedef EvtDataTypeCvt<typename H5Type::XtcType> Super ;
  typedef typename H5Type::XtcType XtcType ;

  // Default constructor
  EvtDataTypeCvtDef ( const std::string& typeGroupName,
                      hsize_t chunk_size,
                      int deflate )
    : EvtDataTypeCvt<typename H5Type::XtcType>(typeGroupName, chunk_size, deflate)
    , m_dataCont(0)
  {
  }

  // Destructor
  virtual ~EvtDataTypeCvtDef () {
    delete m_dataCont ;
  }

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hsize_t chunk_size, int deflate,
      const Pds::TypeId& typeId, const O2OXtcSrc& src)
  {
    // make container for data objects
    typename DataCont::factory_type dataContFactory("data", chunk_size, deflate, true);
    m_dataCont = new DataCont(dataContFactory);
  }

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src)
  {
    // check data size
    if ( H5Type::xtcSize(data) != size ) {
      throw O2OXTCSizeException ( ERR_LOC, Super::typeGroupName(), H5Type::xtcSize(data), size ) ;
    }

    H5Type h5data(data);
    m_dataCont->container(group)->append(h5data);
  }

  /// method called when the driver closes a group in the file
  virtual void closeContainers(hdf5pp::Group group) {
    if (m_dataCont) m_dataCont->closeGroup(group);
  }

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > DataCont ;

  // Data members
  DataCont* m_dataCont;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVTDEF_H
