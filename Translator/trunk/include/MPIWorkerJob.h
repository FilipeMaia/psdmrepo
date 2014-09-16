#ifndef TRANSLATOR_MPIWORKERJOB_H
#define TRANSLATOR_MPIWORKERJOB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MPIWorkerJob
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "openmpi/mpi.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/XtcFilesPosition.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace Translator {

  /**
   * @ingroup Translator
   *
   * @brief class to keep track of worker jobs for translating calib cycle files, from the
   * perspective of the sender.
   *
   * @see H5MPiTranslateApp for usage of this class
   */
class MPIWorkerJob {
 public:
  /// enum identifies known state of a MPIWorkerJob, from perspective of sender.
  /// other than invalidJob, covers the state of the job in terms of the MPI messages going
  /// back and forth between the sender and the worker.
  ///
  /// invalidJob represents a null job
  typedef enum {invalidJob, notStarted, sentToWorker, receivedByWorker, finishedByWorker} JobState_t;

  /// default constructor, sets state to invalidJob
  MPIWorkerJob(int worldRank, int numWorkers);

  /**
   * @brief create job to translate calib file with start position.
   *
   * After checking arguments, sets state to notStarted.
   *
   * @throw std::runtime_error if startCalibNumber is < 0
   */
  MPIWorkerJob(int worldRank, int numWorkers, int calibCycle, const XtcInput::XtcFilesPosition &pos);

  /// desctructor 
  ~MPIWorkerJob();

  JobState_t state() const { return m_state; }

  /// returns true if this is a valid job.
  bool valid() const;

  int startCalibNumber() const { return m_startCalibNumber; }

  /**
   * @brief for valid job whose state is notStared, performs non-blocking send - advances state.
   *
   * This method should only be called for a valid MPIWorkerJob in the notStarted state. It
   * carries out a nonblocking send to send a message to the given worker - or rank in the 
   * MPI_COMM_WORLD. 
   *
   * After advancing the state, saves the workerSentTo argument for subsequent calls.
   *
   * Message format:
   *   data: - filenames and offsets - that the MPIWorkerJob was constructed with. DataType for
   *           a filename/offset pair given by filePosDtype()
   *   tag:  - the starting calib cycle number
   * 
   * @throw std::runtime_error if not valid MPIWorkerJob in notStarted state
   *                          if workerToSendTo is invalid
   */
  void iSend(int workerToSendTo);

  /**
   * @brief tests if worker has finished job
   *
   * does non blocking test on messages. 
   *
   * If the MPIWorkerJob state is sentToWorker, tests to see if the message has been received. 
   * If it has, advances state to receivedByWorker.
   *
   * If the state is receivedByWorker (from previous step or this is where we are now), 
   * initiates non-blocking receive for message coming back from worker that job was completed. 
   *
   * Tests the receive request to see if completed, is so, advances state to finsishedByWorker
   * and returns true. 
   *
   * otherwise returns false
   *
   * @throw std::runtime_error - if the JobState is InvalidJob or notStarted or the
   *                             worker argument is not equal to the worker passed to
   *                             iSend()
   */
  bool testForFinished(int worker);

  /**
   * @brief after waiting for worker to receive start message tests if worker finished
   *
   * This works just as testForFinished() except that if the initial state is sentToWorker, 
   * rather than testing to see if the worker has received the initial message to start, a 
   * blocking wait is performed. 
   */
  bool waitForReceiveAndTestForFinished(int worker);

  /**
   * sets the JobState to finished. Makes no check if this is correct. This is to be called 
   * only if one knows the worker has finished the job.
   */
  void setStateToFinished();


  /**
   * returns the MPI_Request for the finished message from the worker for this
   * job. This parameter will only be valid if the JobState has been advanced to
   * receivedByWorker by the testForFinished() or waitForReceiveAndTestForFinished()
   * functions.
   */
  MPI_Request requestFinish() const { return m_requestFinish; };

  /// struct to hold datatype for buffer of filenames/offsets sent to start workers
  /// on calib cycle files
  struct FilePos {
    enum {MAXFILENAME=192};
    char filename[MAXFILENAME];
    off64_t offset;
  };

  /// returns MPI datatype for FilePos
  MPI_Datatype static filePosDtype();

  typedef enum { noBlocking, waitForReceivedByWorkerState} WaitLevel_t ;
 protected:

  /// internal implementation for  testForFinished() or waitForReceiveAndTestForFinished()
  /// functions
  bool waitTestFinishImpl(int workerToSendTo, WaitLevel_t waitLevel);

  /// A MPIWorkerJob has an internal buffer of FilePos. This function fills
  /// it with the content of the XtcInput::XtcFilesPosition pos argument the MPIWorkerJob
  /// was constructed with
  void fillMsgBufferFromPos();

 private:

  int m_worldRank;      /// world rank - used for diagnostic messages

  int m_numWorkers;     /// number of workers in MPI_WORLD_COMM, used for error checking

  JobState_t m_state;   /// internal state of MPIWorkerJob - reflects communication with worker

  int m_startCalibNumber;  /// starting calib number for the job

  XtcInput::XtcFilesPosition m_startCalibPos;  /// files and offsets the MPIWorkerJob was initialized with

  FilePos * m_msgBuffer;        /// the buffer used to send the start message to a worker

  MPI_Request m_requestStart;   /// the request to see if the worker has received the msg to start the job

  MPI_Request m_requestFinish;  /// the request to see if the worker has send the msg saying it finished the job

  int m_workerSentTo;           /// after sending the start message, the worker the message was sent to

}; // class MPIWorkerJob

std::ostream & operator<<(std::ostream &o, const MPIWorkerJob &mpiWorker);

std::ostream & operator<<(std::ostream &o, MPIWorkerJob::JobState_t);

std::ostream & operator<<(std::ostream &o, MPIWorkerJob::WaitLevel_t);

}; // namespace Translator


#endif // TRANSLATOR_MPIWORKERJOB_H
