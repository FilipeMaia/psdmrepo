//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilesPosition
//
// Author List:
//      David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcFilesPosition.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <set>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/DgramList.h"
#include "XtcInput/Exceptions.h"

using namespace XtcInput;

namespace {
  const char *logger = "XtcFilesPosition";
};

XtcFilesPosition::XtcFilesPosition(const std::list<std::string> &fileNames, 
				   const std::list<off64_t> &offsets)
  : m_xtcFileNames(fileNames.begin(), fileNames.end())
  , m_offsets(offsets.begin(), offsets.end())
{
  if (m_xtcFileNames.size() != m_offsets.size()) {
    throw ArgumentException(ERR_LOC, "fileNames.size() != offsets.size()");
  }
  std::set<int> runs;
  for (unsigned idx = 0; idx < m_xtcFileNames.size(); ++idx) {
    XtcFileName &xtcFileName = m_xtcFileNames.at(idx);
    runs.insert(xtcFileName.run());
    int stream = xtcFileName.stream();
    off64_t offset = m_offsets.at(idx);
    if (m_streamToPos.find(stream) != m_streamToPos.end()) {
      MsgLog(logger, error, "stream=" << stream << " already in map");
      throw ArgumentException(ERR_LOC, "stream occurs twice in FilesPosition");
    }
    std::pair<XtcFileName,off64_t> xtcFileOffset(xtcFileName,offset);
    m_streamToPos[stream]=xtcFileOffset;
  }
  if (runs.size() != 1) {
    throw ArgumentException(ERR_LOC, "XtcFilesPosition: number runs != 1");
  }
  m_run = *runs.begin();
}

boost::shared_ptr<XtcFilesPosition> XtcFilesPosition::makeSharedPtrFromEvent(PSEvt::Event &evt) {
  boost::shared_ptr<DgramList> dgListPtr = evt.get();
  if (not dgListPtr) {
    return boost::shared_ptr<XtcFilesPosition>();
  }
  DgramList &dgList = *dgListPtr;
  std::vector<off64_t> offsets = dgList.getOffsets();
  bool allOffsetsValid = true;
  for (unsigned idx = 0; idx < offsets.size(); ++idx) {
    if (offsets.at(idx) < 0) {
      allOffsetsValid = false;
      break;
    }
  }
  if (not allOffsetsValid) return boost::shared_ptr<XtcFilesPosition>();

  std::list<off64_t> offsetsAsList(offsets.begin(), offsets.end());
  std::vector<XtcInput::XtcFileName> xtcFileNames = dgList.getFileNames();
  std::list<std::string> fileNames;
  for (unsigned idx = 0; idx < xtcFileNames.size(); ++idx) {
    fileNames.push_back(xtcFileNames.at(idx).path());
  }
  return boost::make_shared<XtcInput::XtcFilesPosition>(fileNames, offsetsAsList);
}

std::pair<XtcFileName,off64_t> XtcFilesPosition::getChunkFileOffset(int stream) const 
{ 
  if (not hasStream(stream)) {
    MsgLog(logger,error,"stream " << stream << " not included in file position streams");
    throw ArgumentException(ERR_LOC, "getChunkFileOffset: invalid stream argument");
  }
  return m_streamToPos.find(stream)->second; 
}

std::vector<std::string> XtcFilesPosition::fileNames() const {
  std::vector<std::string> result;
  for (unsigned idx = 0; idx < m_xtcFileNames.size(); ++idx) {
    result.push_back(m_xtcFileNames[idx].path());
  }
  return result;
}
