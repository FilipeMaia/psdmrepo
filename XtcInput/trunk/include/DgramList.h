#ifndef XTCINPUT_DGRAMLIST_H
#define XTCINPUT_DGRAMLIST_H

#include <vector>

#include "XtcInput/Dgram.h"

namespace XtcInput {

/**
 *  @ingroup XtcInput
 *  
 *  @brief Class to hold list of Pds::Dgram's placed into the event store.
 *
 *  The primary purpose of this is to hide the full C++ name frome EventKeys.
 *  
 *  @author David Schneider
 */

class DgramList {
public:
  typedef std::vector<XtcInput::Dgram::ptr> DgramListImpl;

  typedef std::vector<XtcInput::XtcFileName> FileListImpl;

  DgramListImpl getDgrams() const { return m_dgramList; };
  
  FileListImpl getFileNames() const { return m_filenameList; };
  
  size_t size() const { return m_dgramList.size(); }
  
  void push_back(const XtcInput::Dgram & dg);

  Dgram::ptr frontDg() const;

 private:
  DgramListImpl m_dgramList;
  FileListImpl m_filenameList;
};
   
}; // namespace

#endif

