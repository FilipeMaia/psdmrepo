#ifndef O2OTRANSLATOR_CONFIGDATATYPECVT_H
#define O2OTRANSLATOR_CONFIGDATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigDataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <cassert>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OXtcSrc.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

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

namespace O2OTranslator {

template <typename H5Type>
class ConfigDataTypeCvt : public DataTypeCvt<typename H5Type::XtcType> {
public:

  typedef typename H5Type::XtcType XtcType ;

  /**
   *  Constructor for converter.
   *
   *  @param[in] group          HDF5 group inside which we create our stuff
   *  @param[in] typeGroupName  Name of the group for this type, arbitrary string usually
   *                            derived from type, should be unique.
   *  @param[in] src            Data source
   */
  ConfigDataTypeCvt(hdf5pp::Group group, const std::string& typeGroupName, const Pds::Src& src)
    : DataTypeCvt<typename H5Type::XtcType>()
    , m_typeGroupName(typeGroupName)
    , m_group()
  {
    // check if the group already there
    const std::string& srcName = boost::lexical_cast<std::string>(src);
    if ( group.hasChild(typeGroupName) ) {
      hdf5pp::Group typeGroup = group.openGroup(typeGroupName);
      if (typeGroup.hasChild(srcName)) {
        m_group = typeGroup.openGroup(srcName);
        MsgLog("ConfigDataTypeCvt", trace, "group " << typeGroupName << '/' << srcName << " already exists") ;
        return;
      }
    }

    // create separate group
    m_group = group.createGroup(typeGroupName + "/" + srcName);
  }

  // Destructor
  virtual ~ConfigDataTypeCvt () {}

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage )
  {
    // check data size
    if (H5Type::xtcSize(data) != size) {
      if ( size == 0 ) {
        MsgLog("ConfigDataTypeCvt", warning, "Zero XTC payload in " << m_typeGroupName << ", expected size " <<H5Type::xtcSize(data)) ;
        return;
      }
      throw O2OXTCSizeException(ERR_LOC, m_typeGroupName, H5Type::xtcSize(data), size);
    }
    
    // store the data
    H5Type::store(data, m_group);
  }

protected:

private:

  // method called to fill void spaces for missing data
  virtual void fillMissing(const Pds::TypeId& typeId,
                           const O2OXtcSrc& src,
                           const H5DataTypes::XtcClockTimeStamp& time,
                           Pds::Damage damage)
  {
    // For configuration objects we do not do anything if data is missing/damaged
  }

  // Data members
  std::string m_typeGroupName;
  hdf5pp::Group m_group;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGDATATYPECVT_H
