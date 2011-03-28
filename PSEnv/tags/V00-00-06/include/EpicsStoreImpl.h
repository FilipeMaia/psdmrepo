#ifndef PSENV_EPICSSTOREIMPL_H
#define PSENV_EPICSSTOREIMPL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsStoreImpl.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/epics.ddl.h"
#include "psddl_psana/EpicsLib.h"
#include "PSEnv/Exceptions.h"
#include "PSTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEnv {

/**
 *  Class that provides implementation for the EPICS PV store.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class EpicsStoreImpl  {
public:

  // Default constructor
  EpicsStoreImpl () ;

  // Destructor
  ~EpicsStoreImpl () ;

  /// Store EPICS PV
  void store(const boost::shared_ptr<Psana::Epics::EpicsPvHeader>& pv);
  
  /// Get the list of PV names
  void pvNames(std::vector<std::string>& pvNames) const ;

  /// Get CTRL object for given EPICS PV name
  template <typename T>
  boost::shared_ptr<T> getCtrl(const std::string& name) const {
    boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> ptr = getCtrlImpl(name);
    return boost::dynamic_pointer_cast<T>(ptr);
  }
  
  /// Get TIME object for given EPICS PV name
  template <typename T>
  boost::shared_ptr<T> getTime(const std::string& name) const {
    boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> ptr = getTimeImpl(name);
    return boost::dynamic_pointer_cast<T>(ptr);
  }

  /// Get base class object for given EPICS PV name
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> getAny(const std::string& name) const ;

  /**
   *   Get status info for the EPIVS PV.
   *   
   *   @param[in] name      PV name
   *   @param[out] status   EPICS status value
   *   @param[out] severity EPICS severity value
   *   @param[out] time     Time of the last change, can be (0) if time is unknown
   *   
   *   @throw ExceptionEpicsName  if the name of the PV is not known
   */
  void getStatus(const std::string& name, int& status, int& severity, PSTime::Time& time) const ;

  /**
   *  Get the value of the EPICS PV, convert to requested type.
   *  
   *   @param[in] name     PV name
   *   @param[in] idx      value index (for array PVs use non-zero index)
   *   @return the value of the PV
   *   
   *   @throw ExceptionEpicsName  if the name of the PV is not known
   *   @throw ExceptionEpicsConversion  if the PV value cannot be converted to requested type
   */
  template <typename T>
  T getValue(const std::string& name, unsigned idx=0) const
  {
    boost::shared_ptr<Psana::Epics::EpicsPvHeader> pv = getAny(name);
    if (not pv.get()) throw ExceptionEpicsName(ERR_LOC, name);
    try {
      return Psana::EpicsLib::getEpicsValue<T>(*pv, idx);
    } catch (const std::exception& ex) {
      throw ExceptionEpicsConversion(ERR_LOC, name, typeid(T), ex.what());
    }
  }

protected:

  /// Get the index of the PV
  int findIndex(const std::string& name) const;

  // Implementation of the getCtrl which returns generic pointer
  boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> getCtrlImpl(const std::string& name) const;

  // Implementation of the getTime which returns generic pointer
  boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> getTimeImpl(const std::string& name) const;
  
private:

  /// Type for mapping from PV name to PV id
  typedef std::map<std::string, int> Name2ID;
  
  /// Type for mapping from PV ID to EpicsPvCtrl* objects
  typedef std::map<int, boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> > CrtlMap;

  /// Type for mapping from PV ID to EpicsPvTime* objects
  typedef std::map<int, boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> > TimeMap;
  
  // Data members
  Name2ID m_name2id;
  CrtlMap m_ctrlMap;
  TimeMap m_timeMap;

};

} // namespace PSEnv

#endif // PSENV_EPICSSTOREIMPL_H
