//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class H5MpiTranslateApp.cpp
//
// Author List:
//      David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Translator/H5MpiTranslateApp.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <memory>
#include <math.h>
#include <boost/make_shared.hpp>
#include <boost/lexical_cast.hpp>
//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgFormatter.h"
#include "MsgLogger/MsgLogger.h"
#include "psana/PSAna.h"
#include "psana/Exceptions.h"
#include "ConfigSvc/ConfigSvc.h"
#include "PSEvt/ProxyDict.h"
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"
#include "XtcInput/DgramList.h"
#include "XtcInput/XtcFilesPosition.h"
#include "Translator/LoggerNameWithMpiRank.h"
#include "Translator/H5MpiSplitScanDefaults.h"
#include "XtcInput/SharedFile.h"
#include "XtcInput/DgHeader.h"
#include "XtcInput/XtcChunkDgIter.h"

#define MSGLOGLVL info
#define FASTLOGLVL info

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

const char *loggerApp = "H5MpiTranslateApp";
LoggerNameWithMpiRank loggerMaster("H5MpiTranslateApp-Master");
LoggerNameWithMpiRank loggerWorker("H5MpiTranslateApp-Worker");

void mpiGetWorld( int &worldSize, int &worldRank, std::string &processorName) {
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  
  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  
  // Get the name of the processor
  char _processorName[MPI_MAX_PROCESSOR_NAME];
  int nameLen;
  MPI_Get_processor_name(_processorName, &nameLen);
  processorName = std::string(_processorName);
}

  double timeDiffSeconds(const LusiTime::Time & endTime, const LusiTime::Time & startTime) {
    double res = (endTime.sec() - startTime.sec()) + (endTime.nsec()-startTime.nsec())/1e9;
    return res;
  }

  boost::shared_ptr<XtcInput::DgHeader> readDgHeader(int64_t knownOffset, const std::string & knownPath) {
    unsigned liveTimeout = 120; // we really don't need live time out here, this is only called when psana has already read in the datagram
    XtcInput::XtcChunkDgIter chunkDgIter(XtcInput::XtcFileName(knownPath), liveTimeout);
    return chunkDgIter.nextAtOffset(knownOffset);
  }

  bool dgHeaderMatch(Pds::Dgram & pdsDg, uint32_t * buffer) {
    Pds::Dgram & candidateDg = *(Pds::Dgram *)buffer;
    if (pdsDg.seq.clock().nanoseconds() != candidateDg.seq.clock().nanoseconds()) return false;
    if (pdsDg.seq.clock().seconds() != candidateDg.seq.clock().seconds()) return false;
    if (pdsDg.seq.stamp().fiducials() != candidateDg.seq.stamp().fiducials()) return false;
    if (pdsDg.seq.stamp().ticks() != candidateDg.seq.stamp().ticks()) return false;
    if (pdsDg.seq.service() != candidateDg.seq.service()) return false;
    if (pdsDg.seq.type() != candidateDg.seq.type()) return false;
    if (pdsDg.seq.isExtended() != candidateDg.seq.isExtended()) return false;
    if (pdsDg.seq.isEvent() != candidateDg.seq.isEvent()) return false;
    if (pdsDg.env.value() != candidateDg.env.value()) return false;
    
    
    return true;
  }

  bool endsWith(const std::string &a, const std::string &ending) {
    if (a.length() < ending.length()) return false;
    return (0 == a.compare(a.length()-ending.length(), ending.length(), ending));
  }

  int64_t findDg(boost::shared_ptr<XtcInput::DgHeader> knownDg, int64_t candidateOffset, 
                 std::string otherPath, int64_t halfRange, int stepIdx) {
    XtcInput::Dgram::ptr pdsDgPtr = knownDg->dgram();
    if (not pdsDgPtr) throw std::runtime_error("findDg: knownDg is null");
    if ((std::abs(candidateOffset) % 4) != 0) throw std::runtime_error("findDg: known offset is not a multiple of 4");
    Pds::Dgram &pdsDg = *pdsDgPtr;

    // identify a region to look at in otherFile, around candidateOffset
    XtcInput::XtcFileName otherXtc(otherPath);
    int64_t startOffset = std::max(int64_t(0), candidateOffset - halfRange);
    int64_t endOffset = std::max(int64_t(0), candidateOffset + halfRange);
    if ((endOffset - startOffset) < int64_t(sizeof(Pds::Dgram))) return -1;
    const int64_t cutOffForFirst = 0x200000;
    if ((candidateOffset < cutOffForFirst) and (otherXtc.chunk()==0)) {
      endOffset = cutOffForFirst + 0x1000;
    }
    int64_t bytesToRead = endOffset - startOffset + 1;

    // open other file, seek to start of this region. 
    unsigned liveTimeout = 300;
    XtcInput::SharedFile otherSharedFile(otherXtc, liveTimeout);
    struct stat fileStat;
    otherSharedFile.stat(&fileStat);
    unsigned timeWaited = 0;
    unsigned sleepTime = 1;
    const std::string INPROGRESS = ".inprogress";
    bool isInProgress = otherSharedFile.path().extension() == INPROGRESS;
    std::string otherPathNotInProgress(otherPath);
    if (isInProgress) {
      otherPathNotInProgress.erase(otherPath.size()-INPROGRESS.length());
    }
    bool changedFromInProgressToNot = false;

    while ((fileStat.st_size < off_t(startOffset)) and (timeWaited < liveTimeout) and isInProgress) {
      sleep(sleepTime);
      timeWaited += sleepTime;
      if (isInProgress) {
        int fd = open(otherPathNotInProgress.c_str(), O_RDONLY|O_LARGEFILE);
        if (fd >= 0) {
          isInProgress = false;
          changedFromInProgressToNot= true;
          close(fd);
          otherPath = otherPathNotInProgress;
          otherXtc = XtcInput::XtcFileName(otherPath);
          otherSharedFile = XtcInput::SharedFile(otherXtc,0);
        }
      }
      otherSharedFile.stat(&fileStat);
    }
    if (fileStat.st_size < off_t(startOffset)) {
      if (isInProgress) {
        if (not (timeWaited >= liveTimeout)) MsgLog(loggerMaster, error, "unexpected: findDg, isInProgress=True, file is smaller than start, BUT liveTimeOut did not occur, how'd we exit loop?");
        MsgLog(loggerMaster, error, "timeout of " << timeWaited 
               << " sec waiting for file " << otherXtc << " to reach len=" << startOffset);
        throw std::runtime_error("findDg: live timeout");
      } else {
        MsgLog(loggerMaster, info, "findDg: unable to search region because file ends before starting guess. Dg should be earlier in file."
               << "changedFromInProgressToNot=" << changedFromInProgressToNot);
      }
      return -1;
    }
    otherSharedFile.seek(startOffset, SEEK_SET);

    // read in the region, or until end of file
    std::auto_ptr<std::vector<uint8_t> > bufferVec( new std::vector<uint8_t>(bytesToRead) );
    uint8_t *buffer = &(bufferVec->at(0));
    ssize_t bytesRead = otherSharedFile.read((char *)buffer, bytesToRead);
    if (bytesRead < bytesToRead) {
      MsgLog(loggerMaster, info, "findDg: read " << bytesToRead - bytesRead << " fewer bytes than requested. Shortening search space (file was not as long, must be towards the end of a file)");
    }
    // search through the region
    uint32_t * curr = (uint32_t *)buffer;
    uint32_t * last = (uint32_t *)(buffer + bytesRead - 4);
    int64_t foundOffset = -1;

    unsigned spots = 0;
    std::string foundString = "*NOT FOUND*";
    while (curr < last) {
      if (dgHeaderMatch(pdsDg, curr)) {
        foundOffset = startOffset + int64_t((curr - ((uint32_t*)buffer))*4);
        foundString = "*found*";
        break;
      }
      spots++;
      curr++;
    }

    MsgLog(loggerMaster, FASTLOGLVL, "findDg: stepIdx=" << stepIdx << " Dgram: service=" << pdsDg.seq.service() 
           << " type=" << pdsDg.seq.type()
           << " isExtended=" << pdsDg.seq.isExtended()
           << " isEvent=" << pdsDg.seq.isEvent()
           << " nano=0x" << std::hex << pdsDg.seq.clock().nanoseconds() 
           << " sec=0x" << std::hex << pdsDg.seq.clock().seconds()
           << " fid=0x" << std::hex << pdsDg.seq.stamp().fiducials()
           << " ticks=0x" << std::hex << pdsDg.seq.stamp().ticks()
           << " looked at " << spots << " spots around offset 0x" << std::hex << candidateOffset
           << " using interval [" << startOffset << ","  << endOffset << "] "
           << " in " << otherPath << "  " << foundString
           << " inProgress->not: " << changedFromInProgressToNot);

    // return result, -1 or the found offset
    return foundOffset;
  }

  boost::shared_ptr<XtcInput::XtcFilesPosition> fastIndexFindOtherStreamStarts(int streamInSet, 
                                                                               boost::shared_ptr<XtcInput::XtcFilesPosition>beginStepPos, 
                                                                               const std::vector<int> & fastIndexOtherStreams,
                                                                               int stepIdx,
                                                                               int MBforHalfRangeBlock,
                                                                               int numBlocksToTry) {
    if (streamInSet != 0) throw std::runtime_error("fastIndexFindOtherStreamStarts: streamInSet != 0");

    // identify the known offset, we should have daq stream 0 in here
    XtcInput::XtcFilesPosition & filesPos = *beginStepPos;
    std::pair<XtcInput::XtcFileName, int64_t> chunkFileOffset = filesPos.getChunkFileOffset(streamInSet);
    XtcInput::XtcFileName knownXtcFile = chunkFileOffset.first;
    int64_t knownOffset = chunkFileOffset.second;
    std::string knownPath = knownXtcFile.path();
    
    // read the dg from this known offset in stream 0
    boost::shared_ptr<XtcInput::DgHeader> knownDg = readDgHeader(knownOffset, knownPath);

    // set up parameters to produce the other stream filenames based on this known file, and the search space
    const std::string marker = "-s00";
    int whereStreamInPath = knownPath.find(marker);
    if (whereStreamInPath < 1) MsgLog(loggerMaster, fatal, "fastIndex failure, couldn't find marker for stream, " << marker << " in path " << knownPath);

    // start lists for returned object.
    std::list<std::string> newFilenames;
    std::list<off_t> newOffsets;

    std::vector<std::string> currentFilenames = beginStepPos->fileNames();
    std::vector<off64_t> currentOffsets = beginStepPos->offsets();
    
    for (unsigned idx = 0; idx < currentFilenames.size(); ++idx) {
      newFilenames.push_back(currentFilenames.at(idx));
      newOffsets.push_back(currentOffsets.at(idx));
    }
                             
    const int64_t halfRange = MBforHalfRangeBlock << 20l;
    if ((halfRange % 4) != 0) throw std::runtime_error("halfRange must be word aligned, 4 byte boundary");
    std::vector<int64_t> candidateOffsets;
    candidateOffsets.push_back(knownOffset);
    int64_t nextLower = knownOffset;
    int64_t nextHigher = knownOffset;
    for (int blockIdx = 0; blockIdx < numBlocksToTry; ++blockIdx) {
      nextLower -= (2*halfRange - 16);
      nextHigher += (2*halfRange - 16);
      candidateOffsets.push_back(nextLower);
      candidateOffsets.push_back(nextHigher);
    }
    
    for (unsigned streamIdx = 0; streamIdx < fastIndexOtherStreams.size(); ++streamIdx) {
      int otherStream = fastIndexOtherStreams.at(streamIdx);
      if ((otherStream < 0) or (otherStream == streamInSet) or (otherStream > 9)) {
        MsgLog(loggerMaster, fatal, "fastIndex failure, otherstream is " << otherStream 
               << " must be a single digit not equal to streamInSet=" << streamInSet);
      }
      std::string otherPath(knownPath);
      otherPath[whereStreamInPath+3] = '0' + otherStream;
      int64_t otherOffset = -1;
      for (unsigned candIdx = 0; candIdx < candidateOffsets.size(); ++candIdx) {
        int64_t candOffset = candidateOffsets.at(candIdx);
        otherOffset = findDg(knownDg, candOffset, otherPath, halfRange, stepIdx);
        if (otherOffset >=0) break;
      }
      if (otherOffset < 0) {
        MsgLog(loggerMaster, error, "fastIndexFindOtherStreamStarts: failed to find calib cycle after trying " 
               << candidateOffsets.size());
        throw std::runtime_error("fastIndex failure. Could not find calib cycle.");
      }
      newFilenames.push_back(otherPath);
      newOffsets.push_back(off64_t(otherOffset));
    }

    return boost::make_shared<XtcInput::XtcFilesPosition>(newFilenames, newOffsets);
  }

}; // local namespace

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace Translator {

H5MpiTranslateApp::H5MpiTranslateApp (const std::string& appName, char *env[])
  : psana::PSAnaApp( appName ), m_env(env)
{}

H5MpiTranslateApp::~H5MpiTranslateApp ()
{}

int
H5MpiTranslateApp::preRunApp ()
{
  // add logger to error/warning/fatal formatters
  const char* fmt = "[%(level):%(time):%(logger)] %(message)" ;
  const char* errfmt = "[%(level):%(time):%(file):%(line):%(logger)] %(message)" ;
  const char* trcfmt = "[%(level):%(time):%(logger)] %(message)" ;
  const char* dbgfmt = errfmt ;
  MsgLogger::MsgFormatter::addGlobalFormat ( fmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::debug, dbgfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::trace, trcfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::warning, errfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::error, errfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::fatal, errfmt ) ;
  
  return 0;
}

int
H5MpiTranslateApp::runApp ()
{
  mpiGetWorld(m_worldSize, m_worldRank, m_processorName);
  if (m_worldSize < 2) {
    MsgLog(loggerApp,error,"MPI worldsize must be at least 2");
    return -1;
  }
  
  m_numWorkers = m_worldSize - 1;
  m_masterRank = m_worldSize - 1;
  
  m_startTime = LusiTime::Time::now();
  m_eventTime = 0.0;
  m_numCalibJobs = 0;

  std::string cfgFile;
  std::map<std::string, std::string> options;
  setConfigFileAndOptions(cfgFile, options);
  
  // return 0 on success, other values for error (like main())
  int retValue = -1;
  bool abort = false;
  try {
    if (m_worldRank == m_masterRank) {
      retValue = runAppMaster(cfgFile, options);
    } else {
      retValue = runAppWorker(cfgFile, options);
    }
  } catch (const std::exception &e) {
    if (m_worldRank == m_masterRank) {
      MsgLog(loggerMaster, error, "runAppMaster exception: " << e.what());
    } else {
      MsgLog(loggerWorker, error, "runAppWorker exception: " << e.what());
    }
    abort = true;
  } catch (...) {
    if (m_worldRank == m_masterRank) {
      MsgLog(loggerMaster, error, "runAppMaster unknown exception");
    } else {
      MsgLog(loggerWorker, error, "runAppWorker unknown exception");
    }
    abort = true;
  }
  
  if (abort) MPI_Abort(MPI_COMM_WORLD, 2);
  
  m_endTime = LusiTime::Time::now();
  double totalTime = timeDiffSeconds(m_endTime,m_startTime);
  
  
  if (m_worldRank == m_masterRank) {
    MsgLog(loggerMaster,info,"Total time: " << std::setprecision(0) << totalTime
           << " (sec)=" << std::setprecision(2) << totalTime/60.0
           << " (min). Time to read input: " 
           << std::setprecision(2) << 100.0 * m_eventTime/totalTime 
           << "% of total time. Started " << m_numCalibJobs 
           << " worker jobs.");
  } else {
    MsgLog(loggerWorker,info,"Processed " << m_numCalibJobs 
           << " worker jobs in " << std::setprecision(2) << 100.0 * m_eventTime/totalTime << "% of total time");
  }
  
  return retValue;
}

int H5MpiTranslateApp::runAppMaster(std::string cfgFile, std::map<std::string, std::string> options) {
  
  // set options for master
  options["Translator.H5Output.split"]="MPIMaster";
  options["psana.modules"]="";
  
  // Instantiate framework
  PSAna fwk(cfgFile, options);
  
  // get options for reading input
  Context::context_t currentContext = Context::get();
  if (currentContext == 0) {
    MsgLog(loggerMaster, error, "context has not been set");
    return 2;
  }
  ConfigSvc::ConfigSvc cfgSvc(currentContext);
  m_minEventsPerWorker = cfgSvc.get("Translator.H5Output","min_events_per_calib_file",
                                    MIN_EVENTS_PER_CALIB_FILE_DEFAULT);
  m_numEventsToCheckForDoneWorkers = cfgSvc.get("Translator.H5Output",
                                                "num_events_check_done_calib_file",
                                                NUM_EVENTS_CHECK_DONE_CALIB_FILE_DEFAULT);

  m_fastIndex = cfgSvc.get("Translator.H5Output", "fast_index", false);
  m_fastIndexMBhalfBlock = cfgSvc.get("Translator.H5Output", "fi_mb_half_block", 12);
  m_fastIndexNumberBlocksToTry = cfgSvc.get("Translator.H5Output", "fi_num_blocks",3);

  if (m_fastIndex) {
    m_minEventsPerWorker = int(round(m_minEventsPerWorker/6.0));
    m_numEventsToCheckForDoneWorkers = int(round(m_numEventsToCheckForDoneWorkers/2.0));
    MsgLog(loggerMaster,info, "fast_index is true. master sees 1/6 of total events. set master parameters minEventsPerWorker to " << m_minEventsPerWorker
           << " and numEventsToCheckForDoneWorkers to " << m_numEventsToCheckForDoneWorkers);
  }
  if (cfgSvc.get("Translator.H5Output","printenv",false)) {
    WithMsgLog(loggerMaster, info, str) {
      str << "------ Begin Environment -------" << std::endl;
      char *envVar = *m_env;
      int envIdx = 1;
      while (envVar != NULL) {
        str << envVar << std::endl;
        envVar = m_env[envIdx];
        envIdx += 1;
      }
      str << "------ End Environment ------" << std::endl;
    }
  }
  
  if (0 != fwk.modules().size()) {
    MsgLog(loggerMaster,error,"failed to clear psana modules");
    return 2;
  }
  
  // set the input, construct dataSource
  std::vector<std::string> input = inputDataSets();
  std::vector<int> fastIndexOtherStreams;
  int fastStreamInSet = -1;
  if (m_fastIndex) {
    if (input.size() != 1) MsgLog(loggerMaster, fatal, "input size is not one for fastIndex");
    if (input[0].find(":stream") != std::string::npos) MsgLog(loggerMaster, fatal, "fastIndex but input already has stream in it");
    std::string & dset = input[0];
    dset += ":stream=0,80-85";
    fastStreamInSet = 0;
    for (int otherStream =1; otherStream < 6; ++otherStream) {
      fastIndexOtherStreams.push_back(otherStream);
    }
  }
  DataSource dataSource = fwk.dataSource(input);
  if (dataSource.empty()) {
    return 2;
  }
  
  // set initial state of all workers to not valid jobs - i.e.: available
  m_workerJobInProgress.resize(m_numWorkers);
  for (int worker = 0; worker < m_numWorkers; ++worker) {
    m_workerJobInProgress.at(worker)=boost::make_shared<MPIWorkerJob>(m_worldRank, m_numWorkers);
  }
  
  // initialize/declare book-keeping variables to be ued when we go through the data
  // and index where the calib cycles are
  int runIdx = -1;
  int stepIdx = -1;
  int64_t totalEvents = 0;
  bool calibStartsNewJob = true;
  int64_t lastCalibEventStart = 0;
  Run run;
  Step step;
  RunIter runIter = dataSource.runs();
  
  // create H5Output psana module to write master file
  H5Output h5OutputModule("Translator.H5Output");
  
  // to call H5Output methods directly, we need the environment
  PSEnv::Env & env = dataSource.env();
  
  // to call H5Output endCalibCycle, endRun, and endJob methods, we need an empty event
  // since the actual event the module methods would receive is not available throuth 
  // runIter and stepIter
  boost::shared_ptr<AliasMap> amap = env.aliasMap();
  boost::shared_ptr<PSEvt::Event> emptyEvent = 
    boost::make_shared<PSEvt::Event>(boost::make_shared<PSEvt::ProxyDict>(amap));

  MsgLog(loggerMaster,MSGLOGLVL, "starting event loop");
  LusiTime::Time eventStartTime = LusiTime::Time::now();
  LusiTime::Time stepStartTime = LusiTime::Time::now();
  LusiTime::Time stepEndTime = LusiTime::Time::now();
  int64_t stepNumberOfEvents = 0;
  double messagingTimeDuringEventLoop = 0.0;
  bool doBeginJob = true;
  while (true) {
    // loop through runs
    { // scope the call to nextWithEvent to destroy the Event objects after we no longer need them
      std::pair<psana::Run, boost::shared_ptr<PSEvt::Event> > runEvt = runIter.nextWithEvent();
      run = runEvt.first;
      if (not run) break;
      runIdx += 1;
      if (runIdx >= 1) {
        MsgLog(loggerMaster,error,"splitScan cannot process more than one run. " 
               << " The external calib filenames will collide");
        break;
      }
      boost::shared_ptr<PSEvt::Event> evt = runEvt.second;
      if (doBeginJob) {
        // unfortunately, the event associated with beginJob (the Configure transition) is not 
        // available so we use the event associated with the first run.
        h5OutputModule.beginJob(*evt, env);
        doBeginJob = false;
      }
      h5OutputModule.beginRun(*evt, env);
    } // end scope - destroy beginRun event
    stepIdx = -1;
    boost::shared_ptr<XtcInput::XtcFilesPosition> beginStepPos;
    StepIter stepIter = run.steps();
    while (true) {
      // loop through steps in run
      {	// scope the step event
        std::pair<psana::Step, boost::shared_ptr<PSEvt::Event> > stepEvt = stepIter.nextWithEvent();
        step = stepEvt.first;
        if (not step) break;
        stepIdx += 1;
        boost::shared_ptr<PSEvt::Event> evt = stepEvt.second;
        beginStepPos = XtcInput::XtcFilesPosition::makeSharedPtrFromEvent(*evt);
        if (not beginStepPos) {
          MsgLog(loggerMaster, error,"no step position found for step " << stepIdx);
          return -1;
        }
        std::string fastMsg = "";
        if (m_fastIndex) {
          int numDaqStreams = 0;
          for (unsigned stream = 0; stream < 20; ++stream) {
            if (beginStepPos->hasStream(stream)) numDaqStreams++;
          }
          if (numDaqStreams != 1) throw std::runtime_error("expected only 1 DAQ stream in stepPos");
          beginStepPos = fastIndexFindOtherStreamStarts(fastStreamInSet, beginStepPos, fastIndexOtherStreams, 
                                                        stepIdx, m_fastIndexMBhalfBlock, m_fastIndexNumberBlocksToTry);
          fastMsg = " (but fast_index=True, so we are reading only one stream, only need 20hz for real time)";
        }
        if (stepIdx > 0) {
          stepEndTime = LusiTime::Time::now();
          double timeThisStep = timeDiffSeconds(stepEndTime, stepStartTime);
          stepNumberOfEvents = totalEvents - lastCalibEventStart;
          stepStartTime = stepEndTime;
          double eventsPerSec = stepNumberOfEvents/timeThisStep;
          MsgLog(loggerMaster, info, "starting step: " << stepIdx << 
                 " rate=" << eventsPerSec << " evts/sec" << fastMsg << 
                 " num_events=" << stepNumberOfEvents << " seconds=" << timeThisStep);
        }
      }	// destroy the step event
      if (calibStartsNewJob) {
        LusiTime::Time messagingTimeStart = LusiTime::Time::now();        
        XtcInput::XtcFilesPosition &pos = *beginStepPos;
        m_jobsToDo.push(boost::make_shared<MPIWorkerJob>(m_worldRank, m_numWorkers, stepIdx, pos));
        m_numCalibJobs += 1;
        StartWorkerRes_t startWorkerResult;
        int numWorkerJobsStarted = 0;
        do {
          startWorkerResult  = tryToStartWorker();
          if (startWorkerResult == StartedWorker) numWorkerJobsStarted++;
          checkOnLinksToWrite(h5OutputModule);
        } while (startWorkerResult == StartedWorker);
        MsgLog(loggerMaster, info, "Tried to start a new worker for step=" << stepIdx
               << " started " << numWorkerJobsStarted << " worker jobs");
        calibStartsNewJob = false;
        lastCalibEventStart = totalEvents;
        LusiTime::Time messagingTimeEnd = LusiTime::Time::now();
        messagingTimeDuringEventLoop += timeDiffSeconds(messagingTimeEnd, messagingTimeStart);
      }
      EventIter eventIter = step.events();
      while (boost::shared_ptr<PSEvt::Event> evt = eventIter.next()) {
        totalEvents += 1;        
        if ((not calibStartsNewJob) and (totalEvents - lastCalibEventStart >= m_minEventsPerWorker)) {
          calibStartsNewJob = true;
        }
        if (totalEvents % m_numEventsToCheckForDoneWorkers == 0) {
          LusiTime::Time messagingTimeStart = LusiTime::Time::now();
          StartWorkerRes_t startWorkerResult;
          do {
            startWorkerResult  = tryToStartWorker();
            checkOnLinksToWrite(h5OutputModule);
          } while (startWorkerResult == StartedWorker);
          LusiTime::Time messagingTimeEnd = LusiTime::Time::now();
          messagingTimeDuringEventLoop += timeDiffSeconds(messagingTimeEnd, messagingTimeStart);
        }
      }
    } // finish steps in run
  } // finish runs
  MsgLog(loggerMaster, info, "messaging time during event loop (sec): " << messagingTimeDuringEventLoop);
  LusiTime::Time eventEndTime = LusiTime::Time::now();
  m_eventTime += timeDiffSeconds(eventEndTime, eventStartTime);
  
  MsgLog(loggerMaster, MSGLOGLVL, "done with event loop");
  // done iterating through the data
  masterPostIndexing(h5OutputModule);
  // now safe to close the currentRun group
  h5OutputModule.endRun(*emptyEvent, env);
  // close the master file
  h5OutputModule.endJob(*emptyEvent, env);
  
  // send the done message to all the workers
  for (int worker = 0; worker < m_numWorkers; ++worker) {
    MsgLog(loggerMaster, MSGLOGLVL, "about to send finish Send(from=" 
           << m_worldRank<< " -> " << worker << ", tag=0, 0 len buffer)");
    MPI_Send(0,0,MPIWorkerJob::filePosDtype(), worker, 0, MPI_COMM_WORLD);
    MsgLog(loggerMaster, MSGLOGLVL, "sent finish Send(from=" 
           << m_worldRank<< " -> " << worker << ", tag=0, 0 len buffer)");
  }
  return 0 ;
}

bool H5MpiTranslateApp::validJobExists() const {
  for (int worker = 0; worker < m_numWorkers; worker++) {
    MPIWorkerJob &wjob = *m_workerJobInProgress.at(worker);
    if (wjob.valid()) {
      return true;
    }
  }
  return false;
  }

bool H5MpiTranslateApp::freeWorkerExists(int *workerPtr) const {
  for (int worker = 0; worker < m_numWorkers; worker++) {
    MPIWorkerJob &wjob = *m_workerJobInProgress.at(worker);
    if (not wjob.valid()) {
      if (workerPtr) *workerPtr = worker;
      return true;
    }
  }
  if (workerPtr) *workerPtr = -1;
  return false;
}

H5MpiTranslateApp::StartWorkerRes_t H5MpiTranslateApp::tryToStartWorker() {
  // look at queue of calib cycles to do
  if (m_jobsToDo.size() == 0) {
    MsgLog(loggerMaster,trace,"tryToStartWorker: no jobs to do");
    return NoJobsToDo;
  }
  int worker = -1;
  bool foundFreeWorker = freeWorkerExists(&worker);
  if (foundFreeWorker) {
    boost::shared_ptr<MPIWorkerJob> wjob = m_jobsToDo.front();
    m_jobsToDo.pop();
    
    // send job to worker
    MsgLog(loggerMaster,debug,"tryToStartWorker: before iSend worker " << worker << " -> " << wjob->startCalibNumber());
    wjob->iSend(worker);
    // record that worker is working on job
    m_workerJobInProgress.at(worker) = wjob;
    MsgLog(loggerMaster,debug,"tryToStartWorker: after iSend worker " << worker << " -> " << wjob->startCalibNumber());
    MsgLog(loggerMaster, info, "started worker on calib cycle: " << wjob->startCalibNumber());
    return StartedWorker;
  } else {
    MsgLog(loggerMaster,debug,"tryToStartWorker: jobs to do but no available worker");
    return NoWorkerFree;
  }
}

void H5MpiTranslateApp::checkOnLinksToWrite(Translator::H5Output &h5output) {
  bool runGroupValid = h5output.currentRunGroup().valid();
  if (not runGroupValid) {
    MsgLog(loggerMaster,debug,"check on links to write. Nothing to do as runGroup is not valid");
    return;
  }
  // For all the workers that are doing jobs, see if they have sent a finished message.
  for (int worker = 0; worker < m_numWorkers; worker++) {
    boost::shared_ptr<MPIWorkerJob> wjob = m_workerJobInProgress.at(worker);
    if (wjob->valid() and wjob->testForFinished(worker)) {
      MsgLog(loggerMaster,info,"checkOnLinksToWriter: worker " << worker 
             << " finished cc: " << wjob->startCalibNumber());
      try {
        addLinksToMasterFile(worker, wjob, h5output);
      } catch (...) { 
        MsgLog(loggerMaster, error, "checkOnLinksToWrite: call to addLinksToMasterFile failed");
        throw; // callExcept;
      }
      // record that worker is ready for a new job
      m_workerJobInProgress.at(worker)=boost::make_shared<MPIWorkerJob>(m_worldRank, m_numWorkers);
    }
  }
}

// returns worker or -1 if no valid jobs exist
int H5MpiTranslateApp::waitForValidFinishedMPIWorkerJob() {
  if (not validJobExists()) {
    MsgLog(loggerMaster, MSGLOGLVL, "waitForValidFinishedMPIWorkerJob: no valid jobs, returning -1");
    return -1;
  }
  std::vector<MPI_Request> finishRequests;
  std::vector<int> finishRequestWorkers;
  for (int worker = 0; worker < m_numWorkers; ++worker) {
    boost::shared_ptr<MPIWorkerJob> wjob = m_workerJobInProgress.at(worker);
    if (wjob->valid()) {
      if (wjob->waitForReceiveAndTestForFinished(worker)) {
        MsgLog(loggerMaster, MSGLOGLVL, "waitForValidFinishedMPIWorkerJob: valid job. returning worker=" << worker << " job=" << *wjob);
        return worker;
      }
      // not finished, but state should be advanced to receivedByWorker
      MPI_Request workerFinishRequest = wjob->requestFinish();
      if (workerFinishRequest == MPI_REQUEST_NULL) {
        MsgLog(loggerMaster,error,"got null finished request for worker " << worker 
               << " even though state is receivedByWorker");
        throw std::runtime_error("unexpected null finish request");
      }
      finishRequests.push_back(workerFinishRequest);
      finishRequestWorkers.push_back(worker);
    }
  }
  if (finishRequests.size() == 0) {
    // can't be, there was at least one valid job, and none are finished?
    throw std::runtime_error("unexpedted: no finish requests");
  }
  std::vector<MPI_Status> statuses(finishRequests.size());
  int requestIndex = -1;
  MPI_Waitany(finishRequests.size(), &finishRequests.at(0), &requestIndex, &statuses.at(0));
  if (requestIndex < 0) throw std::runtime_error("MPI_Waitany did not set the index");
  if (requestIndex >= int(finishRequestWorkers.size())) throw std::range_error("waitForValidFinishedMPIWorkerJob: MPI_Waitany bad index");
  int worker = finishRequestWorkers.at(requestIndex);
  if (finishRequests.at(requestIndex) != MPI_REQUEST_NULL) {
    MsgLog(loggerMaster,warning, "unexpected: MPI_Waitany did not set worker " 
           << worker << " request to null");
  }
  m_workerJobInProgress.at(worker)->setStateToFinished();
  MsgLog(loggerMaster, MSGLOGLVL, "waitForValidFinishedMPIWorkerJob: waited for finish. returning worker=" << worker);
  return worker;
}

void H5MpiTranslateApp::addLinksToMasterFile(int worker, 
                                             boost::shared_ptr<MPIWorkerJob> wjob, 
                                             H5Output &h5Output) {
  MsgLog(loggerMaster, MSGLOGLVL, "master file link cc " 
         << wjob->startCalibNumber() << " from worker: " << worker);
  // get expected name of file produced for this starting calib number
  int calibNumber = wjob->startCalibNumber();
  std::string ccFilePath = h5Output.splitScanMgr()->getExtCalibCycleFilePath(calibNumber);
  std::string ccFileBaseName = h5Output.splitScanMgr()->getExtCalibCycleFileBaseName(calibNumber);
  MsgLog(loggerMaster, MSGLOGLVL, "addLinksToMasterFile from cc file: " << ccFilePath);
  // open it and go thorugh all CalibCycle:xxxx groups
  hdf5pp::File ccFile = hdf5pp::File::open(ccFilePath, hdf5pp::File::Read);
  if (ccFile.valid()) {
    hdf5pp::Group ccRootGroup = ccFile.openGroup("/");
    if (ccRootGroup.valid()) {
      hdf5pp::Group masterRunGroup = h5Output.currentRunGroup();
      if (masterRunGroup.valid()) {
        int linksCreated = 0;
        char ccGroupName[128];
        sprintf(ccGroupName,"CalibCycle:%4.4d", calibNumber);
        while (ccRootGroup.hasChild(ccGroupName)) {
          h5Output.splitScanMgr()->createExtLink(ccGroupName, ccFileBaseName, masterRunGroup);
          ++linksCreated;
          ++calibNumber;
          sprintf(ccGroupName,"CalibCycle:%4.4d", calibNumber);
        } 
        if (linksCreated == 0) MsgLog(loggerMaster, error, "no calib cycles found in " << ccFilePath);  
        // check for a CalibStore in the cc file
        if (ccRootGroup.hasChild(H5Output::calibStoreGroupName)) {
          // see if we've already added a CalibStore group to the master
          hdf5pp::Group masterConfigureGroup = h5Output.currentConfigureGroup();
          if (masterConfigureGroup.valid()) {
            if (not masterConfigureGroup.hasChild(H5Output::calibStoreGroupName)) {
              // add link
              h5Output.splitScanMgr()->createExtLink(H5Output::calibStoreGroupName.c_str(),
                                                     ccFileBaseName, masterConfigureGroup);
            }
          } else {
            MsgLog(loggerMaster,error,"no valid configure group in master");
          }
        }
      } else {
        MsgLog(loggerMaster, error, "master file run group is not valid during addLinks");
      }
      ccRootGroup.close();
    } else {
      MsgLog(loggerMaster, error, "ccFile " << ccFilePath << " root group is invalid");
    }
    ccFile.close();
  } else {
    MsgLog(loggerMaster, error, "ccFile " << ccFilePath << " is invalid");
  }
}

void H5MpiTranslateApp::masterPostIndexing(Translator::H5Output &h5output) {
  while ((m_jobsToDo.size()>0) or (validJobExists())) {
    
    // first get all idle workers and any remaining jobs
    StartWorkerRes_t startRes = tryToStartWorker();
    if (startRes == StartedWorker) {
      
      continue;
    }
    
    if ((startRes == NoJobsToDo) and (not validJobExists())) {
      // the while loop should prevent us from getting here, we shouldn't get here, 
      MsgLog(loggerMaster, warning, "unexpected: masterPostIndexing loop reached noJobsToDo and no jobs in progress");
      break;
    }
    
    // two cases now:
    // if NoJobsToDo and validJobExists() - wait for job to finish
    // if startRes == NoWorkerFree        - wait for job to finish  
    //    second case implies there are jobs waiting to be assigned
    int finishedWorker = waitForValidFinishedMPIWorkerJob();
    if (finishedWorker < 0) {
      throw std::runtime_error("internal error: there should have been a worker in progress");
    }
    boost::shared_ptr<MPIWorkerJob> finishedJob = m_workerJobInProgress.at(finishedWorker);
    try {
      addLinksToMasterFile(finishedWorker, finishedJob, h5output);
    } catch (const hdf5pp::Hdf5CallException &callExcept) {
      MsgLog(loggerMaster, error, "masterPostIndexing: call to addLinksToMasterFile failed. finishedJob: " << *finishedJob);
      throw callExcept;
    }
    m_workerJobInProgress.at(finishedWorker) = boost::make_shared<MPIWorkerJob>(m_worldRank, m_numWorkers);
    // the next time through the loop, this freed worker will be assigned to 
    // a remaining job
  }
}

/**
 * workers continually check for a message from the indexer.
 * if it is the end message, they are done, otherwise, they run the
 * Translator to write that calib cycle file.
 */
int H5MpiTranslateApp::runAppWorker(std::string cfgFile, std::map<std::string, std::string> options) {
  
  MPI_Datatype fileDtype = MPIWorkerJob::filePosDtype();
  const int MAX_STREAMS = 128;
  MPIWorkerJob::FilePos filePos[MAX_STREAMS];
  int serverRank = m_worldSize - 1;
  int loop = -1;         // for diagnostic messages
  while (true) {
    loop += 1;
    MPI_Status status;
    MsgLog(loggerWorker,MSGLOGLVL,"loop=" << loop << " about to call Recv(to=" << m_worldRank 
           << ", from=" << serverRank << " anytag)");
    MPI_Recv(filePos, MAX_STREAMS, fileDtype, serverRank, MPI_ANY_TAG, 
             MPI_COMM_WORLD, &status);
    int receivedCount;
    MPI_Get_count(&status, fileDtype, &receivedCount);
    MsgLog(loggerWorker,MSGLOGLVL,"loop=" << loop << " worker=" << m_worldRank 
           << " called Recv. recv file count=" << receivedCount);
    if ( receivedCount == 0)  {
      // the done message
      break;
    }
    if ( receivedCount == MAX_STREAMS ) {
      MsgLog(loggerWorker, error, "Received maximum number of streams");
      return -2;
    }
    int startCalibCycle = status.MPI_TAG;
    LusiTime::Time eventStartTime =  LusiTime::Time::now();
    try {
      workerTranslateCalibCycle(cfgFile, options, startCalibCycle, filePos, receivedCount);
      m_numCalibJobs += 1;
    } catch (...) {
      MsgLog(loggerWorker,error,"exception raised during workerTranslateCalibCycle startCalibCycle=" 
             << startCalibCycle);
      throw;
    }
    LusiTime::Time eventEndTime =  LusiTime::Time::now();
    m_eventTime += timeDiffSeconds(eventEndTime, eventStartTime);
    
    // tell server we are done
    MsgLog(loggerWorker,MSGLOGLVL,"runAppWorker:  loop=" << loop << " worker " 
           << m_worldRank << " about to call MPI_Send that done with " << startCalibCycle);
    MPI_Send(0,0,MPI_INT, serverRank, startCalibCycle, MPI_COMM_WORLD);
    MsgLog(loggerWorker,MSGLOGLVL,"runAppWorker:  loop=" << loop << " worker " 
           << m_worldRank << " returned from MPI_Send for " << startCalibCycle);
  }
  return 0;
}

void H5MpiTranslateApp::workerTranslateCalibCycle(std::string cfgFile, 
                                                  std::map<std::string, std::string> options,
                                                  int startCalibCycle, 
                                                  const MPIWorkerJob::FilePos *filePos, 
                                                  int filePosLen)
{
  MsgLog(loggerWorker, MSGLOGLVL, "worker=" << m_worldRank << " translate calib cycle: " << startCalibCycle);
  std::stringstream jmpOffsetOption, jmpFilenameOption;
  for (int idx = 0; idx < filePosLen; ++idx) {
    jmpOffsetOption << filePos[idx].offset;
    jmpFilenameOption << std::string(filePos[idx].filename);
    if (idx < filePosLen-1) {
      jmpOffsetOption << " ";
      jmpFilenameOption << " ";
    }
  }
  
  options["PSXtcInput.XtcInputModule.third_event_jump_offsets"]=jmpOffsetOption.str();
  options["PSXtcInput.XtcInputModule.third_event_jump_filenames"]=jmpFilenameOption.str();
  options["Translator.H5Output.split"]="MPIWorker";
  options["Translator.H5Output.first_calib_cycle_number"]=boost::lexical_cast<std::string>(startCalibCycle);
  
  PSAna fwk(cfgFile, options);
  
  // make sure Translator.H5Output appears in the module list
  Context::context_t currentContext = Context::get();
  if (currentContext == 0) {
    MsgLog(loggerWorker, error, "context has not been set");
    return;
  }
  ConfigSvc::ConfigSvc cfgSvc(currentContext);
  std::vector<std::string> modules = cfgSvc.getList("psana","modules", std::vector<std::string>() );
  bool foundTranslatorModule = false;
  for (unsigned idx = 0; idx < modules.size(); ++idx) {
    if (modules.at(idx) == "Translator.H5Output") {
      foundTranslatorModule = true;
      break;
    }
  }
  if (not foundTranslatorModule) {
    throw std::runtime_error("Translator.H5output not found in module list for mpi worker.");
  }
  
  // set input datasource
  std::vector<std::string> input = inputDataSets();
  DataSource dataSource = fwk.dataSource(input);
  std::vector<std::string> modulesFromDataSource = fwk.modules();
  
  // get event iterator
  EventIter iter = dataSource.events();
  
  while (boost::shared_ptr<PSEvt::Event> evt = iter.next()) {
    // go through the events. The Translator.H5Output module will call stop once
    // it gets through enough calib cycles to meet the min_events_per_calib_file option
  }
  
}

} // namespace Translator


