#ifndef PSANA_TIMETOOL_EVENTDUMP_H
#define PSANA_TIMETOOL_EVENTDUMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::EventDump - psana module classes create an instance of 
//         this to put intermediate results in the event store. What this
//         C++ class does must be kept in sync with the Python psana module
//         PlotAnalyze that looks for results based on assumed key strings
//
//------------------------------------------------------------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "PSEvt/Event.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace TimeTool {

class EventDump {
 public:
  EventDump();
  bool doDump() { return m_doDump; }

  /// unless init is called, all other functions do nothing
  void init(const std::string &keyPrefix);

  /// put laser beam status in event (if init has been called, otherwise return)
  void laserBeamStatus(bool nobeam, bool nolaser, PSEvt::Event &evt);

  /// put signal, sideband and reference in event (if init has been called, otherwise return)
  void sigSbRef(const ndarray<const int32_t,1> &sig, const ndarray<const int32_t,1> &sb, const ndarray<const int32_t,1> &ref, PSEvt::Event &evt);

  /// put reference frame in
  void frameRef(const ndarray<double, 2> &arr, PSEvt::Event &evt);

  /// put return reason in event (if init has been called, otherwise return)
  void returnReason(PSEvt::Event &evt, const std::string &reason);

  /// put 1D ndarray in event (if init has been called, otherwise return)
  template <class T>
   void array(const ndarray<T,1> &arr, PSEvt::Event &evt, const std::string &keySuffix) {
    if (not m_doDump) return;
    if (arr.empty()) return;
    if (keySuffix.size()==0) throw std::runtime_error("EventDump::array - keySuffix is zero len");
    boost::shared_ptr< ndarray<T,1> > arrPtr = boost::make_shared<ndarray<T,1> >(arr.data_ptr(), arr.shape());
    evt.put(arrPtr, m_keyPrefix + keySuffix);
  }

  /// put region of interest and projection dimension for a signal in event (if init has been called, otherwise return)
  void arrayROI(const unsigned roi_lo[2], const unsigned ref_hi[2], unsigned pdim, const std::string &arrayName, PSEvt::Event &evt);

 private:
  bool m_doDump;
  std::string m_keyPrefix;
};

}; // namespace TimeTool

#endif
