//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class SplitScanMgr
//
// Author List:
//     David Schneider
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <uuid/uuid.h>
#include <climits>
#include <vector>

#include "boost/filesystem.hpp"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"

#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"
#include "ErrSvc/Issue.h"

#include "Translator/SplitScanMgr.h"
#include "Translator/LoggerNameWithMpiRank.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

#define TRACELVL trace

namespace {

  LoggerNameWithMpiRank logger("SplitScanMgr");

};

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

using namespace Translator;

SplitScanMgr::SplitScanMgr(const std::string &h5filePath, 
                           bool ccInSubDir,
                           SplitMode splitScanMode,
                           int mpiWorkerStartCalibCycle) :
  m_h5filePath(h5filePath) 
  , m_ccInSubDir(ccInSubDir)
  , m_splitScanMode(splitScanMode)
  , m_mpiWorkerStartCalibCycle(mpiWorkerStartCalibCycle)
{
}

bool SplitScanMgr::thisJobWritesMainOutputFile() const {
  switch (m_splitScanMode) {
  case NoSplit: return true;
  case MPIMaster: return true;
  case MPIWorker: return false;
  }
  throw ErrSvc::Issue(ERR_LOC, "m_splitScanMode value is unhandled?");
}


bool SplitScanMgr::createExtLink(const char *linkName,
                                 const char *targetName,
                                 const std::string & extH5File,
                                 hdf5pp::Group & linkGroupLoc) {
  if (not linkGroupLoc.valid()) {
    MsgLog(logger,error,"createExtLink linkName=" << linkName
	   <<" extFilee="<< extH5File << " but group location is invalid");
    throw ErrSvc::Issue(ERR_LOC,"createExtLink - bad group to add link to");
  }
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  hid_t lcpl_id = H5P_DEFAULT;
  hid_t lapl_id = H5P_DEFAULT;
  herr_t err = H5Lcreate_external(extH5File.c_str(), 
                                  targetName,
                                  linkGroupLoc.id(),
                                  linkName,
                                  lcpl_id, 
                                  lapl_id );
  if (err < 0) {
    MsgLog(logger,error, "H5Lcreate_external failed: link/target name=" << linkName
           << " linkGroupLoc: " << linkGroupLoc.name() << " ext h5 file: " << extH5File);
    throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Lcreate_external failed");
  }
  err = H5Fflush(linkGroupLoc.id(), H5F_SCOPE_GLOBAL);
  if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Fflush");

  MsgLog(logger,TRACELVL,"added external link to master file. targetFile=" 
         << extH5File << " targetName=" << targetName 
         << " linkGroupLoc=" << linkGroupLoc << " linkName=" << linkName);
  return true;
}

void SplitScanMgr::createCCSubDirIfNeeded() const {
  if (m_ccInSubDir) {
    boost::filesystem::path h5path(m_h5filePath);
    boost::filesystem::path CCSubDirPath = h5path.parent_path();
    CCSubDirPath /= getCCSubDirBaseName();
    if (not boost::filesystem::is_directory(CCSubDirPath)) {
      bool success = boost::filesystem::create_directory(CCSubDirPath);
      if (not success) {
        MsgLog(logger, error, "createCCSubDirIfNeeded tried to create direcory: " << CCSubDirPath << " but failed");
        throw std::runtime_error("createCCSubDirIfNeeded failed when creating directory");
      }
    }
  }
}

std::string SplitScanMgr::getCCSubDirBaseName() const {
  boost::filesystem::path h5path(m_h5filePath);
  return h5path.stem().string() + std::string("_ccfiles");
}

std::string SplitScanMgr::getExtFilePath(int calibNumber) const {
  boost::filesystem::path h5path(m_h5filePath);
  boost::filesystem::path newh5path = h5path.parent_path();
  if (m_ccInSubDir) {
    newh5path /= getCCSubDirBaseName();
  }
  newh5path /= getExtFileBaseName(calibNumber);
  return newh5path.string();
}

std::string SplitScanMgr::getExtFileForLink(int calibNumber) const {
  std::string relLinkPath;
  if (m_ccInSubDir) {
    relLinkPath = getCCSubDirBaseName() + "/";
  }
  relLinkPath += getExtFileBaseName(calibNumber);
  return relLinkPath;
}

std::string SplitScanMgr::getExtFileBaseName(int calibNumber) const {
  if (calibNumber <= -2) MsgLog(logger, fatal, "getExtFileBaseName received a calibNumber <= -2");
  if (calibNumber == -1) calibNumber = m_mpiWorkerStartCalibCycle;
  
  boost::filesystem::path h5path(m_h5filePath);
  std::string newFileName = h5path.stem().string();
  char ccNumber[128];
  sprintf(ccNumber,"_cc%4.4d",calibNumber);
  newFileName += std::string(ccNumber);
  newFileName += h5path.extension().string();
  return newFileName;
}

std::string SplitScanMgr::splitModeStr(SplitMode splitMode) {
  switch (splitMode) {
  case NoSplit: return std::string("NoSplit");
  case MPIWorker: return std::string("MPIWorker");
  case MPIMaster: return std::string("MPIMaster");
  }
  return std::string("*unknown*");
}
