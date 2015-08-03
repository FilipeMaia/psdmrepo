//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MPIWorkerJob
//
// Author List:
//      David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Translator/MPIWorkerJob.h"
#include "Translator/LoggerNameWithMpiRank.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string.h>
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {
  LoggerNameWithMpiRank logger("MPIWorkerJob");
} // local namespace

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

#define MSGLOGLVL trace

using namespace Translator;

// make null job
MPIWorkerJob::MPIWorkerJob(int worldRank, int lastWorker) 
  : m_worldRank(worldRank)
   , m_numWorkers(lastWorker)
   , m_state(invalidJob)
   , m_startCalibNumber(-1)
   , m_startCalibPos()
   , m_msgBuffer(0) 
   , m_requestStart(MPI_REQUEST_NULL)
   , m_requestFinish(MPI_REQUEST_NULL)
   , m_workerSentTo(-1)
{
}

// make valid job. not sent to worker yet.
MPIWorkerJob::MPIWorkerJob(int worldRank, int lastWorker, int startCalibNumber, const XtcInput::XtcFilesPosition &pos) 
  : m_worldRank(worldRank)
   , m_numWorkers(lastWorker)
   , m_state(invalidJob)
   , m_startCalibNumber(startCalibNumber)
   , m_startCalibPos(pos)
   , m_msgBuffer(0)
   , m_requestStart(MPI_REQUEST_NULL)
   , m_requestFinish(MPI_REQUEST_NULL)
   , m_workerSentTo(-1)
{
  if (m_startCalibNumber < 0) {
    throw std::runtime_error("MPIWorkerJob constructed with calib number < 0, invalid");
  }
  m_state = notStarted;
  fillMsgBufferFromPos();
}

// free start message buffer if needed
MPIWorkerJob::~MPIWorkerJob() 
{
  if (m_msgBuffer) {
    delete m_msgBuffer;
    m_msgBuffer = 0;
  }
}

  /// returns true if this is a valid job.
bool MPIWorkerJob::valid() const { 
  if (m_startCalibNumber < 0) {
    if (m_state == invalidJob) return false;
    else {
      MsgLog(logger,error,"state is " << m_state << " but m_startCalibNumber is not < 0. It is " << m_startCalibNumber);
      throw std::runtime_error("MPIWorkerJob internal state");
    }
  } 
  // m_startCalibNumber >= 0
  if (m_state == invalidJob) {
    MsgLog(logger,error,"state is " << m_state << " but m_startCalibNumber is >=0. It is " << m_startCalibNumber);
    throw std::runtime_error("MPIWorkerJob internal state");
  }
  return true;
}

// static function - the MPI data type for the start message buffer
MPI_Datatype MPIWorkerJob::filePosDtype() {
  static bool firstCall = true;
  static MPI_Datatype filePosDtypeMPI;
  if (not firstCall) return filePosDtypeMPI;

  firstCall = false;

  int blockcounts[2]={FilePos::MAXFILENAME, 1};
  MPI_Datatype types[2];
  MPI_Aint displacements[2];
  MPI_Datatype filePosType, filePosTypeAfterResize;
  FilePos fpArray[2];
  MPI_Aint nextPos;

  MPI_Get_address(&(fpArray[0]), &displacements[0]);
  MPI_Get_address(&(fpArray[0].offset), &displacements[1]);
  MPI_Get_address(&(fpArray[1]),&nextPos);
  nextPos -= displacements[0];
  displacements[1] -= displacements[0];
  displacements[0]=0;

  types[0]=MPI_CHAR;
  int sizeMpiLong;
  MPI_Type_size(MPI_LONG, &sizeMpiLong);
  if (sizeMpiLong != sizeof(off64_t)) {
    throw std::runtime_error("MPI size of MPI_LONG is not equal to sizeof(off64_t)");
  }
  types[1]=MPI_LONG;

  MPI_Type_create_struct(2, blockcounts, displacements, types, &filePosType);

  // record any gap for arrays of FilePos so we can send an array of them through MPI
  MPI_Type_create_resized(filePosType,0,nextPos,&filePosTypeAfterResize);

  filePosDtypeMPI = filePosTypeAfterResize;
  MPI_Type_commit(&filePosDtypeMPI);
  return filePosDtypeMPI;
}

void MPIWorkerJob::fillMsgBufferFromPos() {
  int numberOfStreams = int(m_startCalibPos.size());
  if (m_msgBuffer) delete m_msgBuffer;
  m_msgBuffer = new FilePos[numberOfStreams];
  std::vector<off64_t> offsets = m_startCalibPos.offsets();
  std::vector<std::string> filenames = m_startCalibPos.fileNames();
  for (int idx=0; idx < numberOfStreams; ++idx) {
    off64_t offset = offsets.at(idx);
    std::string fname = filenames.at(idx);
    if (fname.size() >= FilePos::MAXFILENAME-1) {
      MsgLog(logger,error,"Filename: " << fname << " is longer than " 
	     << FilePos::MAXFILENAME-1 << " characters");
      throw std::runtime_error("filename exceeds max length");
    }
    strncpy(m_msgBuffer[idx].filename, fname.c_str(), FilePos::MAXFILENAME);
    m_msgBuffer[idx].offset = offset;
  }
}

void MPIWorkerJob::iSend(int workerToSendTo) {
  if ((not m_msgBuffer) or (m_state != notStarted) or \
      (m_requestStart != MPI_REQUEST_NULL) or \
      (m_requestFinish != MPI_REQUEST_NULL)) {
    throw std::runtime_error("iSend: internal state wrong");
  }
  if ((workerToSendTo < 0) or (workerToSendTo >= m_numWorkers)) {
    throw std::runtime_error("iSend: invalid workerToSendTo");
  }
  int bufferLen = int(m_startCalibPos.size());
  MsgLog(logger,MSGLOGLVL,"about to call iSend(from=" << m_worldRank
	 << " to=" << workerToSendTo << ",tag=" << m_startCalibNumber << ")");
  MPI_Isend((void *)m_msgBuffer, 
	    bufferLen,
	    filePosDtype(),
	    workerToSendTo, 
	    startCalibNumber(),
	    MPI_COMM_WORLD, 
	    &m_requestStart);
  m_state = sentToWorker;
  m_workerSentTo = workerToSendTo;
  MsgLog(logger,MSGLOGLVL,"called iSend(from=" << m_worldRank
	 << " to=" << workerToSendTo << ",tag=" << m_startCalibNumber << ")");
}

bool MPIWorkerJob::testForFinished(int worker) {
  return waitTestFinishImpl(worker, noBlocking);
}

bool MPIWorkerJob::waitForReceiveAndTestForFinished(int worker) {
  return waitTestFinishImpl(worker, waitForReceivedByWorkerState);
}


bool MPIWorkerJob::waitTestFinishImpl(int worker, WaitLevel_t waitLevel) {
  if (m_state == invalidJob) throw std::runtime_error("MPIWorkerJob::waitTestFinish called on invalid job");
  if (m_state == notStarted) throw std::runtime_error("MPIWorkerJob::waitTestFinish called on not started job");
  if (worker != m_workerSentTo) {
    MsgLog(logger,error,"waitTestFinish: worker=" << worker << " != " << m_workerSentTo);
    throw std::runtime_error("waitTestFinish: worker mismatch");
  }
  if (m_state == sentToWorker) {
    int processed;
    if (waitLevel == noBlocking) {
      MsgLog(logger,MSGLOGLVL,"waitTestFinish: worker=" << worker << " state=" << m_state << " about to call noBlocking MPI_Test");
      MPI_Test(&m_requestStart, &processed, MPI_STATUS_IGNORE);
    } else if (waitLevel == waitForReceivedByWorkerState) {
      MsgLog(logger,MSGLOGLVL,"waitTestFinish: worker=" << worker << " state=" << m_state << " about to call blocking MPI_Wait");
      MPI_Wait(&m_requestStart, MPI_STATUS_IGNORE);
      processed = true;
    } else {
      throw std::runtime_error("waitTestFinish - waitLevel not understood");
    }
    MsgLog(logger,MSGLOGLVL,"waitTestFinish: worker=" << worker << " state=" << m_state << " waitLevel=" << waitLevel << " called MPI_Test or MPI_Wait. processed=" << processed);
    if (processed) {
      m_state = receivedByWorker;
    }
  }
  if (m_state == receivedByWorker) {
    if (m_requestFinish == MPI_REQUEST_NULL)  {
      MsgLog(logger,MSGLOGLVL,"waitTestFinish: about to call IRecv(from=" << m_worldRank
             << ", to=" << worker << ", tag=" << startCalibNumber() << ")");
      MPI_Irecv(0, 0, MPI_INT, worker, startCalibNumber(), MPI_COMM_WORLD, &m_requestFinish);
      MsgLog(logger,MSGLOGLVL,"waitTestFinish: called IRecv(from=" << m_worldRank
             << ", to=" << worker << ", tag=" << startCalibNumber() << ")");
    }
    int processed;
    MsgLog(logger,MSGLOGLVL,"waitTestFinish: worker=" << worker << "? state=receivedByWorker finish sent");
    MsgLog(logger,MSGLOGLVL,"waitTestFinish: about to call MPI_Test for finish from=" << m_worldRank
           << ", to=" << worker);
    MPI_Test(&m_requestFinish, &processed, MPI_STATUS_IGNORE);
    MsgLog(logger,MSGLOGLVL,"waitTestFinish: called MPI_Test for finish from=" << m_worldRank
           << ", to=" << worker << " processed=" << processed);
    if (processed) {
      m_state = finishedByWorker;
    }
  }
  bool retValue = m_state == finishedByWorker;
  MsgLog(logger,MSGLOGLVL,"waitTestFinish(" << worker <<") returning " << retValue);
  return retValue;
}

void MPIWorkerJob::setStateToFinished() {
  if (m_state == invalidJob or m_startCalibNumber < 0) {
    throw std::runtime_error("setStateToFinished called on invalid job");
  }
  m_state = finishedByWorker;
}

std::ostream & Translator::operator<<(std::ostream &o, const MPIWorkerJob &mpiWorker) {
  o << "mpiworker: state=" << mpiWorker.state() << " valid=" << mpiWorker.valid() 
    << " startCalibNumber=" << mpiWorker.startCalibNumber();
  return o;
}

std::ostream & Translator::operator<<(std::ostream &o, MPIWorkerJob::WaitLevel_t waitLevel) {
  switch (waitLevel) {
  case MPIWorkerJob::noBlocking:
    o << "noBlocking";
    break;
  case MPIWorkerJob::waitForReceivedByWorkerState:
    o << "waitForReceivedByWorkerState";
    break;
  default:
    o << "*invalid";
  }
  return o;
}

std::ostream & Translator::operator<<(std::ostream &o, MPIWorkerJob::JobState_t state) {
  switch (state) {
  case MPIWorkerJob::invalidJob:
    o << "invalidJob";
    break;
  case MPIWorkerJob::notStarted:
    o << "notStarted";
    break;
  case MPIWorkerJob::sentToWorker:
    o << "sentToWorker";
    break;
  case MPIWorkerJob::receivedByWorker:
    o << "receivedByWorker";
    break;
  case MPIWorkerJob::finishedByWorker:
    o << "finishedByWorker";
    break;
  default:
    o << "*unknown*";
    break;
  }
  return o;
}
