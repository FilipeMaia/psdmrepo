#ifndef TRANSLATOR_H5MPITRANSLATEAPP_H
#define TRANSLATOR_H5MPITRANSLATEAPP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description: 
//      H5MpiTranslateApp
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <queue>

//----------------------
// Base Class Headers --
//----------------------
#include "psana/PSAnaApp.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Time.h"
#include "Translator/MPIWorkerJob.h"
#include "Translator/H5Output.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief Application class for mpi translator
 *
 *  Includes methods for running the master process as well as the
 *  workers.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author David Schneider
 */
class H5MpiTranslateApp : public psana::PSAnaApp {
public:

  // Constructor
  explicit H5MpiTranslateApp(const std::string& appName = "h5-mpi-translate", char *env[]=NULL) ;

  // destructor
  virtual ~H5MpiTranslateApp () ;

protected :

  ///  override to control MsgLogger format
  virtual int preRunApp () ;

  ///  Main method which runs the whole application
  virtual int runApp () ;

protected:

  /**
   * @brief runs the master process
   *
   * Carries out the following:
   *
   * @li creates a new PSAna framework based on input arguments. Removes all psana modules
   * from the input chain - this allows for locating the calib cycles quickly.
   *
   * @li creates an instance of Translator.H5Output to manage writing of the master file.
   *
   * @li initializes an empty MPIWorkerJob for each rank that is a worker in the MPI_COMM_WORLD.
   * 
   * @li loops through the data - calls H5Output as neccessary for managing master file.
   *  Identifies calib cycles that start new external calib files. For each such calib cycle,
   *  creates a valid MPIWorkerJob.
   * 
   * @li maintains a queue of MPIWorkerJobs to be done.
   *
   * @li identifies free ranks to do MPIWorkerJobs. sends start messages to these ranks.
   *
   * @li periodically checks for done workers. Adds links to master file when workers are
   *     are done.
   *
   * @li waits for all workers to be done, assigns free workers remaining WorkerJob's in the
   *     queue. Clears queue of calib cycles to do
   *
   * @li sends done message to all workers
   *
   * @param[input] cfgFile name of configuration file to initialize PSAna framework with
   * @param[input] options dictionary of options to initialize PSAna framework with
   * 
   * @return non-zero if there was an error
   */
  int runAppMaster(std::string cfgFile, std::map<std::string, std::string> options);

  /**
   * @brief runs a worker process
   *
   * while idle, workers continually check for a message from the master.
   *  if it is the end message, they are done, otherwise, they run the
   * Translator to write that calib cycle file.
   *
   * @param[input] cfgFile name of configuration file to initialize PSAna framework with
   * @param[input] options dictionary of options to initialize PSAna framework with
   * 
   * @return non-zero if there was an error
   */
  int runAppWorker(std::string cfgFile, std::map<std::string, std::string> options);

  /**
   * @brief produces a translated calib cycle file
   *
   * Carries out the following:
   *
   * @li creates a new PSAna framework based on input arguments. Adds options neccessary to
   * run the H5Output module in MPIWorker mode for the given starting calib cycle.
   *
   * @li iterates over the events in the dataset
   *
   * @param[input] cfgFile name of configuration file to initialize PSAna framework with
   * @param[input] options dictionary of options to initialize PSAna framework with
   * @param[input] startCalibCycle the starting calib cycle for the event at filePos
   * @param[input] filePos array of MPIWorkerJob::FilePos structs giving file/offsets for starting
   *               calib cycle
   * @param[input] filePosLen the number of elements in the filePos array
   *
   * @throw std::runtime_error if Translator.H5Output is not among the input modules
   */
  void workerTranslateCalibCycle(std::string cfgFile, std::map<std::string, std::string> options,
				 int startCalibCycle, const MPIWorkerJob::FilePos *filePos, int filePosLen);

  /// enum for return code of tryToStartWorker()
  typedef enum {NoJobsToDo, NoWorkerFree, StartedWorker} StartWorkerRes_t;

  /// utility function for master. checks on jobs that have not been started. Checks for free
  /// workers. Starts job if there is a match.
  StartWorkerRes_t  tryToStartWorker();

  /// tests to see if workers are finished. If so, adds links to their output files to master
  void checkOnLinksToWrite(Translator::H5Output &h5output);

  /**
   * @brief master function to wait for a mpi worker to finish
   *
   * called after the master goes through the data and it waiting for calib cycles to finish
   * so as to write links in the master file.
   *
   * @return if there is at least one  valid job, waits for one to finish. Returns that worker number.
   *         otherwise, returns -1 if no valid jobs
   */
  int waitForValidFinishedMPIWorkerJob();
  
  void addLinksToMasterFile(int worker, boost::shared_ptr<MPIWorkerJob>, Translator::H5Output &h5Output);

  /** 
   * returns true if there is a worker free to do work
   * optionally returns the worker number 
   */
  bool freeWorkerExists(int *worker=0) const;

  /// master process - returns true if there is a valid job - either a job that needs to be
  /// started by a worker, or a worker job in progress
  bool validJobExists() const;

  /// master process - waits for workers to finish calib cycles, starts free workers on remaining jobs, 
  /// writes links to master file
  void masterPostIndexing(Translator::H5Output &h5output);
  

private:

  int m_worldSize;
  int m_worldRank;
  int m_masterRank;

  std::string m_processorName;
  int m_numWorkers;

  int64_t m_minEventsPerWorker;
  int64_t m_numEventsToCheckForDoneWorkers;

  bool m_fastIndex;
  double m_fastIndexMBhalfBlock;
  int m_fastIndexNumberBlocksToTry;

  LusiTime::Time m_startTime, m_endTime;
  double m_eventTime;  // time iterating through events, as opposed to idle time when waiting for
                       // a message (workers) or waiting for a workers to finish (master)
  int m_numCalibJobs; // for the master - the number of calib jobs farmed out,
                      // for a worker - the number of calib jobs processed
  std::queue< boost::shared_ptr<MPIWorkerJob> > m_jobsToDo;
  std::vector< boost::shared_ptr<MPIWorkerJob> > m_workerJobInProgress;
  char **m_env;     // environment, from main
};

} // namespace Translator

#endif // H5MPITRANSLATEAPP_H
