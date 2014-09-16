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
			   SplitMode splitScanMode,
			   int jobNumber, int jobTotal, 
			   int mpiWorkerStartCalibCycle,
			   bool overwrite, 
			   int fileSchemaVersion)  :  
  m_h5filePath(h5filePath) 
  , m_splitScanMode(splitScanMode)
  , m_jobNumber(jobNumber)
  , m_jobTotal(jobTotal)
  , m_mpiWorkerStartCalibCycle(mpiWorkerStartCalibCycle)
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

bool SplitScanMgr::thisJobWritesMainOutputFile() const {
  if ((m_splitScanMode == NoSplit) or
      (m_splitScanMode == MPIMaster) or 
      ((m_splitScanMode == SplitScan) and (jobNumber() == 0))) {
    return true;
  }
  return false;
}

bool SplitScanMgr::thisJobWritesThisCalib(size_t calibNumber) const {
  if (not splitScanMode()) return true;
  // the events the MPIworker sees are controlled by the driver, so a MPIWorker
  // writes all calib cycles that it sees
  if (m_splitScanMode == MPIWorker) return true;
  if (m_splitScanMode == MPIMaster) return false;

  // it is SplitMode:
  if (calibNumber > INT_MAX) {
    calibNumber = calibNumber % INT_MAX;
  }
  if ((int(calibNumber) % jobTotal()) == jobNumber()) return true;
  return false;
}

bool SplitScanMgr::createCalibFileIfNeeded(size_t calibNumber) {
  if (not thisJobWritesThisCalib(calibNumber)) return false;
  if ((m_splitScanMode != MPIWorker) and (m_splitScanMode != SplitScan)) return false;

  size_t calibIndex = getExtCalibIndex(calibNumber);

  MsgLog(logger, TRACELVL, "createCalibFileIfNeeded calibNumber=" 
	 << calibNumber << " calibIndex=" << calibIndex);

  if (m_extCalib.find(calibIndex) == m_extCalib.end()) {
    ExtCalib extCalib;
    extCalib.file = createCalibCycleFile(calibIndex);
    extCalib.startTime = LusiTime::Time::now();
    m_extCalib[calibIndex] = extCalib;
    return true;
  }
  return false;
}

void SplitScanMgr::closeCalibCycleFile(size_t calibCycle) {
  if (not splitScanMode()) {
    MsgLog(logger, warning, "SplitScanMgr file operation but splitScanMode is FALSE");
  }
  ExtCalib &extCalib = getExtCalib(calibCycle);

  std::map<size_t, hdf5pp::Group>::iterator groupIter;
  for (groupIter = extCalib.groups.begin(); groupIter != extCalib.groups.end(); ++groupIter) {
    hdf5pp::Group &group = groupIter->second;
    if (group.valid()) group.close();
  }
  if (extCalib.configGroup.valid()) extCalib.configGroup.close();
  if (extCalib.file.valid()) extCalib.file.close();
}

hdf5pp::File SplitScanMgr::createCalibCycleFile(size_t calibCycle) {
  std::string h5calibFilename = getExtCalibCycleFilePath(calibCycle);
  hdf5pp::File::CreateMode mode = m_overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);

  hdf5pp::File extCalibFile;
  hdf5pp::PListFileAccess fapl;
  try {
    extCalibFile = hdf5pp::File::create(h5calibFilename, mode, fcpl, fapl);
  } catch (const hdf5pp::Hdf5CallException &failMsg) {
    MsgLog(logger, error, "failure in SplitScanMgr::createCalibCycleFile");
    throw failMsg;
  }
  MsgLog(logger, TRACELVL, "SplitScanMgr::createCalibCycleFile - created " << h5calibFilename);

  // store schema version for this file
  extCalibFile.createAttr<uint32_t>(":schema:version").store(fileSchemaVersion());

  // add attributes specifying schema features
  const char* tsFormat = "full"; // older translator supported a "short" format
  extCalibFile.createAttr<const char*>(":schema:timestamp-format").store(tsFormat) ;
  extCalibFile.createAttr<uint32_t> (":schema:bld-shared-split").store(1);
  extCalibFile.createAttr<uint32_t> (":schema:bld-config-as-evt").store(1);

  // add UUID to the file attributes
  uuid_t uuid ;
  uuid_generate( uuid );
  char uuid_buf[64] ;
  uuid_unparse ( uuid, uuid_buf ) ;
  extCalibFile.createAttr<const char*> ("UUID").store ( uuid_buf ) ;

  // add some metadata to the top group
  extCalibFile.createAttr<const char*> ("origin").store ( "psana-translator" ) ;
  LusiTime::Time startTime = LusiTime::Time::now();
  extCalibFile.createAttr<const char*> ("created").store ( startTime.toString().c_str() ) ;

  return extCalibFile;
}

hdf5pp::Group SplitScanMgr::createConfigureGroupInExtCalibFile(size_t calibCycle) {
  if (not isMPIWorker()) throw std::runtime_error("createConfigureCycleGroupInExtCalibFile but not MPIWorker");
  ExtCalib &extCalib = getExtCalib(calibCycle);
  if (not extCalib.file.valid()) throw std::runtime_error("createConfigureCycleGroupInExtCalibFile file not valid");
  if (extCalib.configGroup.valid()) throw std::runtime_error("createConfigureCycleGroupInExtCalibFile config group already created");
  extCalib.configGroup = extCalib.file.createGroup("config");
  return extCalib.configGroup;
}

hdf5pp::Group SplitScanMgr::createCalibCycleGroupInExtCalibFile(size_t calibCycle) {

  ExtCalib &extCalib = getExtCalib(calibCycle);
  if (extCalib.groups.find(calibCycle) != extCalib.groups.end()) {
    MsgLog(logger, fatal, "group for calib cycle already created");
  }
  
  char groupName[128];
  sprintf(groupName,"CalibCycle:%4.4lu", calibCycle);
  extCalib.groups[calibCycle] = extCalib.file.createGroup(groupName);
  return extCalib.groups[calibCycle];
}

void SplitScanMgr::newExtLnkForMaster(const char *linkName,
				      size_t calibCycle,
				      hdf5pp::Group & linkGroupLoc) {
  if (not (thisJobWritesMainOutputFile() and splitScanMode() and (not isMPIMaster()))) {
    MsgLog(logger, error, "newExtLnkForMaster should only be called for non mpi split mode with master");
    return;
  }
  if (m_masterLnksToWrite.find(calibCycle) != m_masterLnksToWrite.end()) {
    MsgLog(logger, error, "newExtLnkForMaster has already been called for calibCycle: " 
           << calibCycle << " are multiple runs being translated? No entry added.");
    return;
  }
  m_masterLnksToWrite[calibCycle]=MasterLinkToWrite(linkName, linkGroupLoc);
  MsgLog(logger, TRACELVL, "received notice of external link to add to master file, linkName="
         << linkName<< " calibCycle=" << calibCycle << " linkGroupLoc=" << linkGroupLoc);
}

size_t SplitScanMgr::getExtCalibIndex(size_t calibCycle) {
  size_t calibIndex;
  if (m_splitScanMode == MPIWorker) {
    calibIndex = m_mpiWorkerStartCalibCycle;
  } else {
    calibIndex = calibCycle;
  }
  return calibIndex;
}

SplitScanMgr::ExtCalib & SplitScanMgr::getExtCalib(size_t calibCycle) {
  size_t calibIndex = getExtCalibIndex(calibCycle);

  if (m_extCalib.find(calibIndex) == m_extCalib.end()) {
    throw ErrSvc::Issue(ERR_LOC, "calib index not found");
  }
  ExtCalib &extCalib = m_extCalib[calibIndex];
  return extCalib;
}

hdf5pp::Group SplitScanMgr::extCalibFileRootGroup(size_t calibCycle) {
  ExtCalib &extCalib = getExtCalib(calibCycle);
  if (not extCalib.file.valid()) {
    MsgLog(logger, fatal, "cannot return root group, file is not valid");
  }
  hdf5pp::Group rootGroup;
  try {
    rootGroup = extCalib.file.openGroup("/");
  } catch (const hdf5pp::Hdf5CallException &callExcept) {
    MsgLog(logger, error, "SplitScanMgr::extCalibFileRoot - hdf5 call exceptoin on opening file");
    throw callExcept;
  }
  return rootGroup;
}

void SplitScanMgr::updateMasterLinks(enum UpdateExtLinksMode updateMode) {
  if (not isNonMPISplitMaster()) {
    MsgLog(logger, error, "updateMasterLinks - not non MPI split master");
  }
  std::vector<size_t> calibsToErase;
  std::map< size_t, MasterLinkToWrite>::iterator lnks;
  for (lnks = m_masterLnksToWrite.begin(); lnks != m_masterLnksToWrite.end(); ++lnks) {
    size_t calibCycle = lnks->first;
    MasterLinkToWrite & masterLinkToWrite = lnks->second;
    if ((updateMode == writeAll) or 
        ((updateMode == writeFinishedOnly) and calibFileIsFinished(calibCycle))) {
      std::string fullPath = getExtCalibCycleFilePath(calibCycle);
      createExtLink(masterLinkToWrite.linkName.c_str(), 
		    getExtCalibCycleFileBaseName(calibCycle), 
		    masterLinkToWrite.linkGroupLoc);
      calibsToErase.push_back(calibCycle);
    }
  }
  MsgLog(logger, TRACELVL, "updateCalibCycleExtLinks updateMode=" << updateModeToStr(updateMode)
         << " created " << calibsToErase.size() << " external links.");

  for (std::vector<size_t>::iterator ccIter = calibsToErase.begin();
       ccIter != calibsToErase.end(); ++ccIter) {
    m_masterLnksToWrite.erase(*ccIter);
  }
}

bool SplitScanMgr::calibFileIsFinished(size_t calibCycle) {
  if (m_splitScanMode != SplitScan) {
    throw std::runtime_error("must be in splitmode");
  }
  size_t nextCalibCycle = calibCycle + jobTotal();
  std::string nextPath = getExtCalibCycleFilePath(nextCalibCycle);
  return boost::filesystem::exists(nextPath);
}

bool SplitScanMgr::createExtLink(const char *linkName,
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
  const char *targetName = linkName;
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

std::string SplitScanMgr::getExtCalibCycleFilePath(size_t calibCycle) {
  boost::filesystem::path h5path(m_h5filePath);
  boost::filesystem::path newh5path = h5path.parent_path();
  newh5path /= getExtCalibCycleFileBaseName(calibCycle);
  return newh5path.string();
}

std::string SplitScanMgr::getExtCalibCycleFileBaseName(size_t calibCycle) {
  boost::filesystem::path h5path(m_h5filePath);
  std::string newFileName = h5path.stem().string();
  char ccFileName[128];
  sprintf(ccFileName,"_cc%4.4lu", calibCycle);
  newFileName += std::string(ccFileName);
  newFileName += h5path.extension().string();
  return newFileName;
}

std::string SplitScanMgr::updateModeToStr(enum UpdateExtLinksMode mode) {
  if (mode == writeAll) return std::string("writeAll");
  if (mode == writeFinishedOnly) return std::string("writeFinishedOnly");
  return std::string("*unknown*");
}

std::string SplitScanMgr::splitModeStr(SplitMode splitMode) {
  switch (splitMode) {
  case NoSplit: return std::string("NoSplit");
  case SplitScan: return std::string("SplitScan");
  case MPIWorker: return std::string("MPIWorker");
  case MPIMaster: return std::string("MPIMaster");
  }
  return std::string("*unknown*");
}
