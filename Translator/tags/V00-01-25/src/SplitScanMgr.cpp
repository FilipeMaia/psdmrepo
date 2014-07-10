#include <uuid/uuid.h>
#include <climits>
#include <vector>

#include "Translator/SplitScanMgr.h"
#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

#include "boost/filesystem.hpp"

namespace {
  const char * logger = "SplitScanMgr";
};

using namespace Translator;

SplitScanMgr::SplitScanMgr(const std::string &h5filePath, bool splitScanMode,
                     int jobNumber, int jobTotal, bool overwrite,
                     int fileSchemaVersion)  :  
  m_h5filePath(h5filePath) 
  , m_splitScanMode(splitScanMode)
  , m_jobNumber(jobNumber)
  , m_jobTotal(jobTotal)
  , m_overwrite(overwrite)
  , m_fileSchemaVersion(fileSchemaVersion)
{
  if (splitScanMode) {
    if (jobNumber < 0) MsgLog(logger, fatal, "jobNumber < 0: jobNumber = " << jobNumber);
    if (jobTotal < 0) MsgLog(logger, fatal, "jobTotal < 0: jobTotal = " << jobTotal);
    if (jobNumber >= jobTotal) MsgLog(logger, fatal, "jobNumber >= jobTotal, jobNumber= " << jobNumber 
                                      << ", jobTotal=" << jobTotal);
  }
}

bool SplitScanMgr::noSplitOrJob0() const
{
  if (not splitScanMode()) return true;
  if (jobNumber() == 0) return true;
  return false;
}

bool SplitScanMgr::thisJobWritesThisCalib(size_t calibNumber) const {
  if (not splitScanMode()) return true;
  if (calibNumber > INT_MAX) {
    calibNumber = calibNumber % INT_MAX;
  }
  if ((int(calibNumber) % jobTotal()) == jobNumber()) return true;
  return false;
}

void SplitScanMgr::closeCalibCycleFile(size_t calibCycle) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  std::map<size_t, ExtCalib>::iterator calibPos = m_extCalib.find(calibCycle);
  if (calibPos == m_extCalib.end()) {
    MsgLog(logger, fatal, "closeCalibCycleFile called with calibCycle=" << calibCycle 
           << " but this calibCycle does not exist in map");
  }
  ExtCalib &extCalib = calibPos->second;
  extCalib.group.close();
  extCalib.file.close();
  //  m_extCalib.erase(calibPos);
}

hdf5pp::Group SplitScanMgr::createNextCalibCycleFile(size_t calibCycle) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  if (m_extCalib.find(calibCycle) != m_extCalib.end()) {
    MsgLog(logger, fatal, "createNextCalibCycleFile called with calibCycle=" << calibCycle 
           << " but this calibCycle has already been created");
  }
  std::string h5calibFilename = getExtCalibCycleFilePath(calibCycle);
  hdf5pp::File::CreateMode mode = m_overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);

  ExtCalib extCalib;
  hdf5pp::PListFileAccess fapl;
  extCalib.file = hdf5pp::File::create(h5calibFilename, mode, fcpl, fapl);

  // store schema version for this file
  extCalib.file.createAttr<uint32_t>(":schema:version").store(fileSchemaVersion());

  // add attributes specifying schema features
  const char* tsFormat = "full"; // older translator supported a "short" format
  extCalib.file.createAttr<const char*>(":schema:timestamp-format").store(tsFormat) ;
  extCalib.file.createAttr<uint32_t> (":schema:bld-shared-split").store(1);
  extCalib.file.createAttr<uint32_t> (":schema:bld-config-as-evt").store(1);

  // add UUID to the file attributes
  uuid_t uuid ;
  uuid_generate( uuid );
  char uuid_buf[64] ;
  uuid_unparse ( uuid, uuid_buf ) ;
  extCalib.file.createAttr<const char*> ("UUID").store ( uuid_buf ) ;

  // add some metadata to the top group
  extCalib.startTime = LusiTime::Time::now();
  extCalib.file.createAttr<const char*> ("origin").store ( "psana-translator" ) ;
  extCalib.file.createAttr<const char*> ("created").store ( extCalib.startTime.toString().c_str() ) ;
  char groupName[128];
  sprintf(groupName,"CalibCycle:%4.4lu", calibCycle);
  extCalib.group = extCalib.file.createGroup(groupName);
  m_extCalib[calibCycle]=extCalib;
  return extCalib.group;
}

void SplitScanMgr::newCalibCycleExtLink(const char *linkName,
                                 size_t calibCycle,
                                 hdf5pp::Group & linkGroupLoc) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  if (m_masterLnksToWrite.find(calibCycle) != m_masterLnksToWrite.end()) {
    MsgLog(logger, error, "newCalibCycleExtLink has already been called for calibCycle: " 
           << calibCycle << " are multiple runs being translated? No entry added.");
    return;
  }
  m_masterLnksToWrite[calibCycle]=MasterLinkToWrite(linkName, linkGroupLoc);
  MsgLog(logger, trace, "received notice of external link to add to master file, linkName="
         << linkName<< " calibCycle=" << calibCycle << " linkGroupLoc=" << linkGroupLoc);
}

void SplitScanMgr::updateCalibCycleExtLinks(enum UpdateExtLinksMode updateMode) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  std::vector<size_t> calibsToErase;
  std::map< size_t, MasterLinkToWrite>::iterator lnks;
  for (lnks = m_masterLnksToWrite.begin(); lnks != m_masterLnksToWrite.end(); ++lnks) {
    size_t calibCycle = lnks->first;
    MasterLinkToWrite & masterLinkToWrite = lnks->second;
    if ((updateMode == writeAll) or 
        ((updateMode == writeFinishedOnly) and calibFileIsFinished(calibCycle))) {
      createCalibCycleExtLink(masterLinkToWrite.linkName.c_str(), 
                              calibCycle, 
                              masterLinkToWrite.linkGroupLoc);
      calibsToErase.push_back(calibCycle);
    }
  }
  MsgLog(logger, trace, "updateCalibCycleExtLinks updateMode=" << updateModeToStr(updateMode)
         << " created " << calibsToErase.size() << " external links.");

  for (std::vector<size_t>::iterator ccIter = calibsToErase.begin();
       ccIter != calibsToErase.end(); ++ccIter) {
    m_masterLnksToWrite.erase(*ccIter);
  }
}

bool SplitScanMgr::calibFileIsFinished(size_t calibCycle) {
  size_t nextCalibCycle = calibCycle + jobTotal();
  std::string nextPath = getExtCalibCycleFilePath(nextCalibCycle);
  return boost::filesystem::exists(nextPath);
}

bool SplitScanMgr::createCalibCycleExtLink(const char *linkName,
                                        size_t calibCycle,
                                        hdf5pp::Group & linkGroupLoc) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  boost::filesystem::path targetPath(getExtCalibCycleFilePath(calibCycle));
  std::string targetFile = targetPath.filename().string();
  
  hid_t lcpl_id = H5P_DEFAULT;
  hid_t lapl_id = H5P_DEFAULT;
  const char *targetName = linkName;
  herr_t err = H5Lcreate_external(targetFile.c_str(), 
                                  targetName,
                                  linkGroupLoc.id(),
                                  linkName,
                                  lcpl_id, 
                                  lapl_id );
  if (err < 0) {
    MsgLog(logger,error, "H5Lcreate_external failed: link/target name=" << linkName
           << " linkGroupLoc: " << linkGroupLoc.name() << " ext calib filename: " << targetFile);
    throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Lcreate_external failed");
  }
  err = H5Fflush(linkGroupLoc.id(), H5F_SCOPE_GLOBAL);
  if (err < 0) throw hdf5pp::Hdf5CallException(ERR_LOC, "H5Fflush");

  MsgLog(logger,trace,"added external link to master file. targetFile=" 
         << targetFile << " targetName=" << targetName 
         << " linkGroupLoc=" << linkGroupLoc << " linkName=" << linkName);
  return true;
}

std::string SplitScanMgr::getExtCalibCycleFilePath(size_t calibCycle) {
  boost::filesystem::path h5path(m_h5filePath);
  std::string newFileName = h5path.stem().string();
  char ccFileName[128];
  sprintf(ccFileName,"_cc%4.4lu", calibCycle);
  newFileName += std::string(ccFileName);
  newFileName += h5path.extension().string();
  boost::filesystem::path newh5path = h5path.parent_path();
  newh5path /= newFileName;
  return newh5path.string();
}

std::string SplitScanMgr::updateModeToStr(enum UpdateExtLinksMode mode) {
  if (mode == writeAll) return std::string("writeAll");
  if (mode == writeFinishedOnly) return std::string("writeFinishedOnly");
  return std::string("*unknown*");
}
