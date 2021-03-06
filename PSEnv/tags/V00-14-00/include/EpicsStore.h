#ifndef PSENV_EPICSSTORE_H
#define PSENV_EPICSSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsStore.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <boost/enable_shared_from_this.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEnv/EpicsStoreImpl.h"
#include "pdsdata/xtc/Src.hh"
#include "psddl_psana/epics.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  @ingroup PSEnv
 *  
 *  @brief Class implementing storage for EPICS data in psana framework.
 *  
 *  The EPICS store keeps track of all current EPICS value during the event 
 *  loop in the framework. It is updated with the new values whenever 
 *  new EPICS data is read from the input file.
 *
 *  Epics data is retrieved by specifying a name. The EpicsStore first checks
 *  if this name is an alias, if not it is assumed to be a pvName.
 *
 *  When the initial EPICS data is added, the EpicsStore checks for aliases that 
 *  have the same name as a pvName. These aliases are discared (debug messages 
 *  are generated for discared  aliases). This prevents aliases that hide pv's.
 *  A consequence is that users cannot use aliases to swap the names of existing 
 *  epics pv's.
 *
 *  While the same EPICs pv can come from multiple sources, the EpicsStore does
 *  not expose the source to the user in the interface to get EPICs data. When a 
 *  user gets a pv, the EpicsStore returns the last one stored. When the same TIME 
 *  pv is coming from two or more sources during the same event, users will generally
 *  prefer the one with the latest internal time stamp value. When storing EPICS data, 
 *  an optional eventId can be passed. EpicsStore will use this to identify TIME pv's
 *  from the same event. For multiple pv's from the same event, it will then store the
 *  one with the most recent stamp.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Env
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class EpicsStore : public boost::enable_shared_from_this<EpicsStore>, boost::noncopyable {
public:

  /**
   *  Helper class which converts the result of EpicsStore::getPV() call into
   *  real data object. The object of this type can be converted to smart
   *  pointer to one of the Psana::Epics::EpicsPv* classes (defined in psddl_psana
   *  package).
   */
  struct EpicsPV {
    
    // conversion operators
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlHeader>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlString>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlString>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlShort>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlShort>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlFloat>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlFloat>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlEnum>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlEnum>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlChar>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlChar>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlLong>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlLong>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvCtrlDouble>() {
      return m_impl->getCtrl<Psana::Epics::EpicsPvCtrlDouble>(m_name);
    }

    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeHeader>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeString>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeString>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeShort>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeShort>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeFloat>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeFloat>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeEnum>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeEnum>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeChar>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeChar>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeLong>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeLong>(m_name);
    }
    operator boost::shared_ptr<Psana::Epics::EpicsPvTimeDouble>() {
      return m_impl->getTime<Psana::Epics::EpicsPvTimeDouble>(m_name);
    }

    operator boost::shared_ptr<Psana::Epics::EpicsPvHeader>() {
      return m_impl->getAny(m_name);
    }
    
    EpicsStoreImpl* m_impl;
    std::string m_name;
  };
  
  /**
   *  Helper class which converts the result of EpicsStore::get() call into
   *  real data. Objects of this type can be converted to one of the basic
   *  numeric types or std::string.
   */
  struct EpicsValue {
    
    // conversion operators
    operator int() { return m_impl->getValue<int>(m_name, m_idx); }
    operator unsigned() { return m_impl->getValue<unsigned>(m_name, m_idx); }
    operator short() { return m_impl->getValue<short>(m_name, m_idx); }
    operator unsigned short() { return m_impl->getValue<unsigned short>(m_name, m_idx); }
    operator long() { return m_impl->getValue<long>(m_name, m_idx); }
    operator unsigned long() { return m_impl->getValue<unsigned long>(m_name, m_idx); }
    operator long long() { return m_impl->getValue<long long>(m_name, m_idx); }
    operator unsigned long long() { return m_impl->getValue<unsigned long long>(m_name, m_idx); }
    operator char() { return m_impl->getValue<char>(m_name, m_idx); }
    operator signed char() { return m_impl->getValue<signed char>(m_name, m_idx); }
    operator unsigned char() { return m_impl->getValue<unsigned char>(m_name, m_idx); }
    operator float() { return m_impl->getValue<float>(m_name, m_idx); }
    operator double() { return m_impl->getValue<double>(m_name, m_idx); }
    operator std::string() { return m_impl->getValue<std::string>(m_name, m_idx); }
    
    EpicsStoreImpl* m_impl;
    std::string m_name;
    int m_idx;
  };
  
  // Default constructor
  EpicsStore () ;

  // Destructor
  ~EpicsStore () ;

  /**
   *  @brief Store EPICS PV
   *
   *  storing a pv first requires identifying the pvName. If this is not stored in the 
   *  pv header (as with TIME pv's), or it is not passed explicitly through the optional 
   *  argument pvName, then the pvName will be found using the pvid in the pv header and the 
   *  source. If no pvName is found (this can happen for damaged data missing configuration 
   *  events) one will be created based on the src and pvId in the header.
   *
   *  The optional argument eventTag can be set to a value >= 0 to identify TIME pv's from the 
   *  same event. When store gets a TIME pv that is from the same event as the last one stored,
   *  it will only update the internal store if the new pv has a stamp value > than the stamp
   *  of the previously stored pv. If eventTag < 0, this check of the stamp values is not made.
   *
   *  @param[in] pv        pv header
   *  @param[in] src       src pv
   *  @param[in] pvName    optional pvName, overrides use of internal mechanism to find pvName
   *                       based on pvid and src. Note - if passed, this should be a valid pvName 
   *                       consinstent with the data, not an alias.
   *  @param[in] eventTag  optional key for grouping TIME pv's from the same event.
   *                       If eventTag >=0, it is treated as such a key. Can be a simple counter 
   *                       for identifying events - not related to the EventId of a psana Event.
   */
  void store(const boost::shared_ptr<Psana::Epics::EpicsPvHeader>& pv, const Pds::Src& src, 
             const std::string *pvName = NULL, long eventTag = -1) {
    m_impl->store(pv, src, pvName, eventTag);
  }

  /// Store alias name for EPICS PV.
  void storeAlias(const Pds::Src& src, int pvId, const std::string& alias) {
    m_impl->storeAlias(src, pvId, alias);
  }

  /**
   *  @brief  Get the full list of PV names and aliases.
   *
   *  Returned list includes the names of all PVs and aliases.
   */
  std::vector<std::string> names() const
  {
    std::vector<std::string> names;
    m_impl->names(names);
    return names;
  }

  /**
   *  @brief  Get the list of PV names.
   *
   *  Returned list includes the names of all PVs but no alias names.
   */
  std::vector<std::string> pvNames() const 
  {
    std::vector<std::string> names;
    m_impl->pvNames(names);
    return names;
  }

  /**
   *  @brief  Get the list of PV aliases.
   *
   *  Returned list includes the names of all alias names but no PV names.
   */
  std::vector<std::string> aliases() const
  {
    std::vector<std::string> names;
    m_impl->aliases(names);
    return names;
  }

  /**
   *  @brief  Get alias name for specified PV name.
   *
   *  If specified PV is not found or does not have an alias an empty string is returned.
   */
  std::string alias(const std::string& pv) const
  {
    return m_impl->alias(pv);
  }

  /**
   *  @brief  Get PV name for specified alias name.
   *
   *  If specified alias is not found an empty string is returned.
   */
  std::string pvName(const std::string& alias) const
  {
    return m_impl->pvName(alias);
  }

  /**
   *   @brief Get the value for a given PV or alias name.
   *   
   *   @param[in] name      PV name
   *   @param[in] idx       value index (for array PVs)
   *   @return  object that is convertible to regular numeric types or std::string.
   *   
   *   This method does not throw but conversion from EpicsValue to final 
   *   type can throw ExceptionEpicsName or ExceptionEpicsConversion.
   */
  EpicsValue value(const std::string& name, int idx=0) const {
    EpicsValue v = { m_impl.get(), name, idx };
    return v;
  }
  
  /**
   *   @brief Get status information for a given PV or alias name.
   *   
   *   @param[in] name      PV name
   *   @param[out] status   EPICS status value
   *   @param[out] severity EPICS severity value
   *   @param[out] time     Time of the last change, can be (0) if time is unknown
   *   
   *   @throw ExceptionEpicsName  if the name of the PV is not known
   */
  void status(const std::string& name, int& status, int& severity, PSTime::Time& time) const {
    m_impl->getStatus(name, status, severity, time);
  }
  
  /** 
   *  @brief Find EPICS PV given its PV or alias name.
   *  
   *  @param[in] name      PV name
   *  @return  Object convertible to shared_ptr<T> where T is one of the epics PV classes.
   *   
   */
  EpicsPV getPV(const std::string& name) const {
    EpicsPV pv = { m_impl.get(), name };
    return pv;
  }
  
  /**
   *  @brief Access implementation object.
   *
   *  Do not use this method unless you know what you are doing. This is not
   *  supposed to be used by end clients, but may be useful for other services
   *  like Python wrappers. The name is intentionally long to make you very
   *  uncomfortable using it.
   */
  const EpicsStoreImpl& internal_implementation() const { return *m_impl; }

protected:

private:

  // Data members
  boost::scoped_ptr<EpicsStoreImpl> m_impl;  ///< Pointer to implementation.

};

} // namespace PSEnv

#endif // PSENV_EPICSSTORE_H
