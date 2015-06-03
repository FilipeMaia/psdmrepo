#ifndef PSDDL_PDS2PSANA_XTCCONVERTER_H
#define PSDDL_PDS2PSANA_XTCCONVERTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcConverter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <map>
#include <utility>
#include <boost/shared_ptr.hpp>
//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Xtc.hh"
#include "PSEvt/Event.h"
#include "PSEnv/EnvObjectStore.h"
#include "PSEnv/EpicsStore.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/**
 *  @brief Class responsible for conversion of XTC objects into 
 *  real psana objects.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcConverter  {
public:

  // Default constructor
  XtcConverter () ;

  // Destructor
  ~XtcConverter () ;
  
  /**
   *  @brief Convert one object and store it in the event.
   */
  void convert(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore);
  

  /**
   *  @brief Returns list of type_info pointers for types that convert puts in event/config store
   */
  std::vector<const std::type_info *> getConvertTypeInfoPtrs(const Pds::TypeId &) const;

  /**
   *  @brief Convert one object and store it in the config store.
   */
  void convertConfig(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EnvObjectStore& cfgStore);

  /**
   *  @brief Convert one object and store it in the epics store.
   */
  void convertEpics(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore, long epicsStoreTag=-1);

  /**
   *  @brief returns true if typeId is a special xtc that is split into several different xtcs during convert
   */
  bool isSplitType(const Pds::TypeId &typeId) const { return m_sharedSplit.isSplitType(typeId); };

  /**
   *  @brief returns the typeId's of the xtc's that a special shared xtc is split into.
   *  Returns an empty list if typeId is not special.
   */
  std::vector<Pds::TypeId> splitTypes(const Pds::TypeId &typeId) const;
  
protected:
  
private:

  /**
   *  @brief Utility class to compare Pds::TypeId's for map indexing
   *
   *  @author David Schneider
   */
  class LessTypeId {
  public:
    bool operator()(const Pds::TypeId &a, const Pds::TypeId &b) const {
      if (a.id()==b.id()) return (a.version()<b.version());
      return a.id()<b.id();
    }
  };

  /**
   *  @brief Utility class to keeps track of special shared xtc types
   *
   * Class that keeps track of special shared xtc types. For each 
   * shared xtc type, this class maintains a list of sub xtc types 
   * that the shared type should be split into during convert.
   * For each sub xtc, stores the xtc type id, and where the sub xtc
   * should be stored - the Event store or the config store.
   *
   *  @author David Schneider
   */
  class SharedSplitXtcs {
  public:
    /**
     * @brief default constructor creates shared/split xtc data
     */
    SharedSplitXtcs();

    typedef enum {storeToConfig, storeToEvent} StoreTo;
    
    /**
     * @brief returns True if typeid is for a special shared type to be split
     */
    bool isSplitType(const Pds::TypeId &typeId) const;

    typedef std::pair<Pds::TypeId,StoreTo> SplitEntry;
    
    /**
     * @brief returns split types and where to store for special shared type id.
     * Returns empty list if not special shared type id.
     */
    const std::vector<SplitEntry> & splitTypes(const Pds::TypeId &typeId) const;

    /**
     * @brief checks if a type id and storeTo value is correct
     *
     * @param[in] sharedTypeId   type id for special shared type xtc type that is split
     * @param[in] entry          shared types are split into several sub xtc.  Specify
     *                           the first, second, third such sub xtc via a 0-up index
     * @param[in] entryTypeId    the typeId you expect to see for this sub xtc
     * @param[in] entryStoreTo   where you expect this sub xtc to be stored
     *
     * @return true if entryTypeId,entryStoreTo matches the internal information
     *         for the given entry and sharedTypeId
     */
    bool equal(const Pds::TypeId &sharedTypeId, const unsigned entry, 
               const Pds::TypeId &entryTypeId, const StoreTo entryStoreTo);
  private:
    std::map<Pds::TypeId, std::vector< SplitEntry>, LessTypeId > m_sharedSplitMap;
    static const std::vector<SplitEntry> emptyList;
  };  // class SharedSplitXtcs


  // Data members
  SharedSplitXtcs m_sharedSplit;

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_XTCCONVERTER_H
