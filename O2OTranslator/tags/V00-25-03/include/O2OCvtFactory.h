#ifndef O2OTRANSLATOR_O2OCVTFACTORY_H
#define O2OTRANSLATOR_O2OCVTFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OCvtFactory.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/CvtOptions.h"
#include "O2OTranslator/DataTypeCvtI.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace O2OTranslator {
class CalibObjectStore;
class ConfigObjectStore;
class O2OMetaData;
}


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Class which instantiates converters for all known data types.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class O2OCvtFactory : boost::noncopyable {
public:

  typedef boost::shared_ptr<DataTypeCvtI> DataTypeCvtPtr ;
  typedef std::vector<DataTypeCvtPtr> DataTypeCvtList;

  /**
   *  @brief Constructor instantiates converters for all known data types
   */
  O2OCvtFactory(ConfigObjectStore& configStore, CalibObjectStore& calibStore,
      const O2OMetaData& metadata, const CvtOptions& cvtOptions);

  /**
   *  @brief Return the list of converters for given arguments.
   *
   *  Find or create converters (there may be any number of those) for a given
   *  combination of group, type, and source.
   */
  DataTypeCvtList getConverters(const hdf5pp::Group& group, Pds::TypeId typeId, Pds::Src src);

  /**
   *  @brief Notify factory that the group is about to be closed.
   *
   *  When the group is closed all converters associated with the group are
   *  "closed" as well meaning that they are simply destroyed.
   */
  void closeGroup(const hdf5pp::Group& group);

private:

  // Instantiate all converters for given triplet
  DataTypeCvtList makeCvts(const hdf5pp::Group& group, Pds::TypeId typeId, Pds::Src src);

  // helper class for ordering (TypeId, Src) combination
  typedef std::pair<Pds::TypeId, Pds::Src> TypeAndSource;
  struct TypeAndSourceCmp {
    bool operator()(const TypeAndSource& lhs, const TypeAndSource& rhs) const;
  };

  typedef std::map<TypeAndSource, DataTypeCvtList, TypeAndSourceCmp> TypeSrcCvtMap;
  typedef std::map<hdf5pp::Group, TypeSrcCvtMap> GroupCvtMap;
  
  ConfigObjectStore& m_configStore;
  CalibObjectStore& m_calibStore;
  const O2OMetaData& m_metadata;
  CvtOptions m_cvtOptions;
  GroupCvtMap m_groupCvtMap;
  
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OCVTFACTORY_H
