#include <stdexcept>

#include "TimeTool/EventDump.h"

#include "ndarray/ndarray.h"

using namespace TimeTool;

EventDump::EventDump() : m_doDump(false)
{
}

void EventDump::init(const std::string &keyPrefix) 
{
  m_doDump = true;
  m_keyPrefix = keyPrefix;
}

void EventDump::laserBeamStatus(bool nobeam, bool nolaser, PSEvt::Event &evt)
{
  if (not m_doDump) return;
  unsigned shape[1]={2};
  boost::shared_ptr<ndarray<int8_t,1> > data = boost::make_shared< ndarray<int8_t,1> >(shape);
  (*data)[0]=nobeam;
  (*data)[1]=nolaser;
  evt.put(data, m_keyPrefix + "_nobeam_nolaser");
}

void EventDump::sigSbRef(const ndarray<const int32_t,1> &sig, const ndarray<const int32_t,1> &sb, const ndarray<const int32_t,1> &ref, PSEvt::Event &evt)
{
  if (not m_doDump) return;
  array(sig, evt, "_sig");
  array(sig, evt, "_sb");
  array(sig, evt, "_ref");
}

void EventDump::arrayROI(const unsigned roi_lo[2], const unsigned roi_hi[2], unsigned pdim, const std::string & arrayName, PSEvt::Event &evt) {
  if (not m_doDump) return;
  const unsigned shape[1] = {5};
  boost::shared_ptr<ndarray<unsigned,1> > evtData = boost::make_shared< ndarray<unsigned,1> >(shape);
  (*evtData)[0]=roi_lo[0];
  (*evtData)[1]=roi_lo[1];
  (*evtData)[2]=roi_hi[0];
  (*evtData)[3]=roi_hi[1];
  (*evtData)[4]=pdim;
  evt.put(evtData, m_keyPrefix + arrayName + "_roilo_roihi_pdim");
}

void EventDump::returnReason(PSEvt::Event &evt, const std::string &reason) {
  if (not m_doDump) return;
  evt.put(boost::shared_ptr<std::string>(new std::string(reason)), m_keyPrefix + "_return");
}

