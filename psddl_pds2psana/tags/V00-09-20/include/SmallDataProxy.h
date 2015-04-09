#ifndef PSDDL_PDS2PSANA_SMALLDATAPROXY_H
#define PSDDL_PDS2PSANA_SMALLDATAPROXY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SmallDataEventProxy
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include <boost/enable_shared_from_this.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"
#include "PSEvt/ProxyI.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/SharedFile.h"
#include "pdsdata/xtc/Dgram.hh"
#include "psddl_pds2psana/XtcConverter.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/**
 *  @brief implement small data proxy mechanism
 *
 *  @see 
 *
 *  @version $Id$
 *
 *  @author David Schneier
 * 
 * An instance of this class should be created for each datagram in a small data xtc file.
 *
 * Small data files must end with .smd.xtc (the logic that defines this is in the class XtcFileName).
 *
 * A small data file must be in a subdirectory called 'smalldata' to the directory where the large 
 * xtc file is. 
 *
 * Clients of this class should do three things:
 * 
 *  1) use the method makeSmallDataProxy below to make instances - this will return a null pointer
 *     if no smallDataProxy should be made (for instance, the xtc file doesn't end with .smd.xtc
 *  2) call addEventProxy (or addEnvProxy) for each proxied item in the datagram
 *  3) call finalize once the datagram has been processed
 *
 * There are also several utilty functions to encapsulate details about smallDataProxy from the
 * xtc files
 *
 * Below are some implementation details:
 *
 * The makeSmallDataProxy is constructed with the user event, environment, information about live
 * mode, and an optimal read byte value. With each call to addEventProxy or addEnvProxy, it puts
 * a special proxy object in the user event or configStore. This proxy - a distinct class from
 * SmallDataProxy - has a pointer back to the SmallDataProxy, as well as an object ID for the 
 * event data it is proxying. When the user gets this event data, this is what happens:
 * 
 *  1) The special proxy calls the parent SmallDataProxy - passing it its id
 *  2) The SmallDataProxy sees what group this id is in. During the finalize routine, SmallDataProxy
 *     grouped different objects to optimize the reading from disk. That is, if optimalReadBytes
 *     was 3 MB, and the datagram contained 3 back to back 1MB camera frames, they will all be in the
 *     same group. One read operation is done for the group, and all 3 cameras are dispatched - as
 *     described below.
 *  3) If the group has not been loaded, it is read from disk
 *  4) A shared pointer to the xtc for this object is created, sharing ownership with the block of
 *     memory read from disk.
 *  5) This xtc is then dispatched - using the pds2psana DDL back-end generated dispatch routine.
 *  5) Since the backend dispatch routine requires an event to put its result into, and since
 *     we cannot re-use the user event which already has a proxy for this key - the implementation
 *     of SmallDataProxy contains its own internal event to use for the output of dispatch.
 *  6) After dispatch is called, we check the internal largeDataEvent for the result.
 */

class SmallDataProxy : public boost::enable_shared_from_this<SmallDataProxy>, boost::noncopyable  {
public:
  /**
   *  @brief public method to construct object, checks that is small data xtc file
   *
   * Clients of this class should create instances with this function. It will return a null
   * pointer if conditions are incorrect for small data proxy. Currently these conditions are 
   * simply that the xtcfile be of form the xtcfile is not a small data file.
   *
   *  @param[in] xtcFileName   Can be small data file (ending with .smd.xtc) or not - null pointer
   *                           returned if not
   *  @param[in] liveMode      true if large data file to read from will be coming in live mode
   *  @param[in] liveTimeOut   time (in seconds) to wait for the large data file to appear, as well as
   *                           to wait during seek operations into the large data file.
   *  @param[in] cvt           XtcConverter - could be used for identifying split types, final dispatch, etc
   *  @param[in] evt           pointer to user event. For non L1 datagrams, such as beginCalibCycle, pass null.
   *  @param[in] env           reference to user environemnt. 
   *  @param[in] optimalReadBytes  set to approximate size to group multiple objects into - to reduce number
   *                           of reads. For example, if file system reads a minimum of 4MB per read, set to 
   *                           something close to 4MB. Group sizes may be more or less - depending on how
   *                           xtc distributed in datagram.
   * 
   *  @return Shared pointer of the correct type.

   */
  static boost::shared_ptr<psddl_pds2psana::SmallDataProxy> 
    makeSmallDataProxy(const XtcInput::XtcFileName & xtcFileName, bool liveMode, 
                       unsigned liveTimeOut, psddl_pds2psana::XtcConverter &cvt,
                       PSEvt::Event* evt, PSEnv::Env &env, unsigned optimalReadBytes = 3<<20);

  /**
   *  @brief Returns true if this is a type that proxies other types
   */
  static bool isSmallDataProxy(const Pds::TypeId &);

  /**
   * @brief return typeid of proxied type
   */
  static const Pds::TypeId getSmallDataProxiedType(const Pds::Xtc *xtc);

  /**
   *  @brief Returns list of type_info pointers that a proxy type proxies
   */
  static std::vector<const std::type_info *> getSmallConvertTypeInfoPtrs(const Pds::Xtc * xtc,
                                                                         const psddl_pds2psana::XtcConverter &cvt);

  // Destructor
  ~SmallDataProxy();
  
  /**
   *  @brief adds proxy in user event for type in a small data xtc proxy.
   *
   * Throws error if xtc is not for small data proxy.
   */
  void addEventProxy(const boost::shared_ptr<Pds::Xtc>& xtc, 
                     std::vector<const std::type_info *>  typeInfoPtrs);
  
  /**
   *  @brief adds proxy in user configStore in environemnt for type in a small data xtc proxy.
   *
   * Throws error if xtc is not for small data proxy.
   */
  void addEnvProxy(const boost::shared_ptr<Pds::Xtc>& xtc,
                   std::vector<const std::type_info *>  typeInfoPtrs);
  
  
  /**
   *  @brief must be called after going through datagram. groups items.
   */
  void finalize();

  bool isFinalized() { return m_finalized; }

  typedef int ObjId;    // individual objects that we proxy from this event
  typedef int GroupId;  // group id, we may read a number of individual objects at once

  boost::shared_ptr<void> get(ObjId objId, const PSEvt::EventKey &eventKey);

 protected:
  // constructor
  SmallDataProxy(const XtcInput::XtcFileName &xtcFileName, bool liveMode, 
                 unsigned liveTimeOut, psddl_pds2psana::XtcConverter &cvt,
                 boost::shared_ptr<PSEvt::Event> evt, 
                 boost::shared_ptr<PSEnv::Env> env, unsigned optimalReadBytes = 3<<20);

  bool addProxyChecksFailed(const boost::shared_ptr<Pds::Xtc>& xtc, psddl_pds2psana::XtcConverter &cvt);

  ObjId getNextObjId() { return m_nextObjId++; }

  /// string dumping values of object, primarily for debugging
  std::string dumpStr();
  
  /// loads the given group and dispatches all its objects
  void loadGroup(GroupId groupId);

  /// reads given block from possible live file - returning pointer to raw data
  boost::shared_ptr<int8_t> readBlockFromLargeFile(int64_t startOffset, int64_t extent);

 private:

  //----- helper classes
  struct ObjData {
    ObjId objId;
    GroupId groupId;
    int64_t fileOffset;
    uint32_t extent;
    Pds::TypeId typeId;  // store only to validate large xtc
    Pds::Src src;        // store only to validate large xtc
    Pds::Damage damage;  // store only to validate large xtc
    // constructors
    ObjData(ObjId id, int64_t fo, uint32_t ext, Pds::TypeId tp, Pds::Src sc, Pds::Damage dmg) 
      : objId(id), groupId(-1), fileOffset(fo), extent(ext), typeId(tp), src(sc), damage(dmg) {};
    ObjData() : objId(-1), groupId(-1), fileOffset(-1), extent(0) {};
  };

  class LessObjDataByFileOffset {
  public:
    bool operator()(ObjData * const a, ObjData * const b) const {
      return (a->fileOffset) < (b->fileOffset);
    }
  };

  struct GroupData {
    std::vector<ObjId> ids;  // these ids must be orderd by FileOffset, first to last in group
    bool loaded;
    GroupData() : loaded(false) {};
    GroupData(const std::vector<ObjId> &_ids) : ids(_ids), loaded(false) {};
   };
  //-------- end helper classes

  XtcInput::XtcFileName m_smallXtcFileName;
  XtcInput::XtcFileName m_largeXtcFileName;
  
  bool m_liveMode;
  unsigned m_liveTimeOut;
  psddl_pds2psana::XtcConverter &m_cvt;
  bool m_finalized;
  unsigned m_optimalReadBytes;
  bool m_triedToOpen;
  XtcInput::SharedFile m_largeFile;
  boost::weak_ptr<PSEvt::Event> m_evtForProxies;
  std::auto_ptr<PSEvt::Event> m_evtForLargeData;
  boost::weak_ptr<PSEnv::EnvObjectStore> m_configStoreForProxies;


  ObjId m_nextObjId;
  std::map<ObjId, ObjData> m_ids;
  std::map<GroupId, GroupData > m_groups;
};
 
} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_SMALLDATAPROXY_H
