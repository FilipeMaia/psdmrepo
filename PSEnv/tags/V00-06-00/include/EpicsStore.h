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
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Env
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class EpicsStore : boost::noncopyable {
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
   *  numeric types of std::string.
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

  /// Store EPICS PV, will add new PV or update existing PV.
  void store(const boost::shared_ptr<Psana::Epics::EpicsPvHeader>& pv, const Pds::Src& src) {
    m_impl->store(pv, src);
  }

  /// Get the list of PV names, all known names are returned.
  std::vector<std::string> pvNames() const 
  {
    std::vector<std::string> names;
    m_impl->pvNames(names);
    return names;
  }

  /**
   *   @brief Get the value for a given PV name
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
   *   @brief Get status information for a given PV name.
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
   *  @brief Find EPICS PV given its name.
   *  
   *  @param[in] name      PV name
   *  @return  Object convertible to shared_ptr<T> where T is one of the epics PV classes.
   *   
   */
  EpicsPV getPV(const std::string& name) const {
    EpicsPV pv = { m_impl.get(), name };
    return pv;
  }
  
protected:

private:

  // Data members
  boost::scoped_ptr<EpicsStoreImpl> m_impl;  ///< Pointer to implementation.

};

} // namespace PSEnv

#endif // PSENV_EPICSSTORE_H
