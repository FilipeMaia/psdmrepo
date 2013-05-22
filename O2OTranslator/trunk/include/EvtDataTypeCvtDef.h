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
#include "O2OTranslator/CvtOptions.h"
#include "O2OTranslator/O2OExceptions.h"

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
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
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

  /**
   *  Constructor for converter
   *
   *  @param[in] group          Group in HDF5 file
   *  @param[in] typeGroupName  Name of the group for this type, arbitrary string usually
   *                            derived from type, should be unique.
   *  @param[in] src            Data source
   *  @param[in] cvtOptions     Conversion options
   *  @param[in] schemaVersion  Schema version number
   *  @param[in] dsname         Dataset name, usually it is "data", may be changed to anything
   */
  EvtDataTypeCvtDef ( const hdf5pp::Group& group,
                      const std::string& typeGroupName,
                      const Pds::Src& src,
                      const CvtOptions& cvtOptions,
                      int schemaVersion,
                      const std::string& dsname = "data")
    : EvtDataTypeCvt<typename H5Type::XtcType>(group, typeGroupName, src, cvtOptions, schemaVersion)
    , m_dsname(dsname)
    , m_dataCont()
  {
  }

  // Destructor
  virtual ~EvtDataTypeCvtDef () {}

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
  {
    // make container for data objects
    m_dataCont = Super::template makeCont<DataCont>(m_dsname, group, true);
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

    // this is guaranteed to be called after makeContainers
    H5Type h5data(data);
    m_dataCont->append(h5data);
  }

  // fill containers for missing data
  virtual void fillMissing(hdf5pp::Group group,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src)
  {
    // this is guaranteed to be called after makeContainers
    m_dataCont->resize(m_dataCont->size() + 1);
  }

private:

  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;

  // Data members
  const std::string m_dsname;  ///< Dataset name
  boost::shared_ptr<DataCont> m_dataCont;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVTDEF_H
