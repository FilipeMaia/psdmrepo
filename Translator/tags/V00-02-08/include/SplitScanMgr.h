#ifndef TRANSLATOR_SPLITSCANMGR_H
#define TRANSLATOR_SPLITSCANMGR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SplitCanMgr
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "LusiTime/Time.h"

namespace Translator {

  /**
   *  @ingroup Translator
   *  
   *  @brief Manages split scan operations.
   * 
   * In split scan mode, calib cycles are written into separate files by potentially different
   * jobs. This class manages these operations. 
   *
   * The main writer classes, H5Output and EpicsH5GroupDirectory can share an instance of SplitScanMgr
   * and use it to determine if they are responsible for writing a calib cycle or event. H5Output
   * will call SplitScanMgr when CalibCycles start and end. H5Output will use methods of SplitScanMgr 
   * to create the external files with calib cycle data, and to update the master file with the 
   * external links.
   *
   * When H5Output discovers a new calib cycle in the data, these methods should be called:
   * 
   * createNextCalibCycleFile - create the external file for the next calib cycle 
   *                            and return a hdf5pp:Group to add events to. This should be called
   *                            by the job that writes that calib cycle.
   *
   * newCalibCycleExtLink - This should be called by job0 - or the MPI master driver. The master job
   *                        is soley responsible for manipulating the master file with links to the calib 
   *                        cycle files. This routine does not neccessarily add the link to the master file 
   *                        when called. Ideally we only add a link after the calib cycle file is finished 
   *                        (so as to eliminate errors that programs will get get by following links to 
   *                        unfinished, or non existant files).
   *
   * updateCalibCycleExtLinks - this is called to inform SplitScanMgr to update the external links.
   *
   * Presently, newCalibCycleExtLink immediately adds the link and updateCalibCycleExtLinks does nothing.
   * In the future we may add logic to only add valid links.
   *
   *  @version $Id$
   *
   *  @author David Schneider

   */
  class SplitScanMgr {
  public:

    enum SplitMode { NoSplit, SplitScan, MPIWorker, MPIMaster };
    static std::string splitModeStr(SplitMode splitMode);

    /**
     *  @brief constructor for SplitScanMgr
     *
     *  @param[in] h5filePath   - full path name for the master file.
     *                            All calib cycle files will be created in the same directory.
     *  @param[in] splitScanMode - a SplitScan value
     *  @param[in] linkGroupLoc - the group where the link is made.
     *
     *  Typically linkGroupLoc will be for Run:0000 in the master file, and if calibNumber is
     *  for example 3, linkName will be CalibCycle:0003. If the master filename is output.h5,
     *  the filename linked to will be output_cc0003.h5. It is always a relative path in the 
     *  current directory.
     */
    SplitScanMgr(const std::string &h5filePath, 
		 SplitMode splitScanMode,
		 int jobNumber, int jobTotal, 
		 int mpiWorkerStartCalibCycle,
		 bool overwrite,
		 int fileSchemaVersion);

    /// return true if responsible for main h5 file
    bool thisJobWritesMainOutputFile() const;

    /// returns true if operating in split scan mode - MPI or otherwise
    bool splitScanMode() const { return m_splitScanMode != NoSplit; }

    /// returns true if MPIWorker
    bool isMPIWorker() const { return m_splitScanMode == MPIWorker; }

    /// returns true if master for splitScan - non MPI
    bool isNonMPISplitMaster() const { return (m_splitScanMode == SplitScan) \
                                     	       and (jobNumber()==0); }

    /// returns true if MPIMaster
    bool isMPIMaster() const { return m_splitScanMode == MPIMaster; }

    /// this job number (0 if not split scan mode)
    int jobNumber() const { return m_jobNumber; }

    /// total number of jobs (1 if not split scan mode)
    int jobTotal() const { return m_jobTotal; }

    /// hdf5 file schema version
    int fileSchemaVersion() const { return m_fileSchemaVersion; }

    /**
     *  @brief returns true if this job writes this calib cycle
     *
     * always returns true if not split scan. For split scan, a job
     * writes a calib cycle if
     *
     *         calibNumber % jobTotal == jobNumber
     *  
     * For example, if jobTotal = 3, and jobNumber = 1, then this job writes calib cycles
     *  1, 4, 7, ...
     *
     *  @param[in] calibNumber   calib cycle number (0 up) to see if this job writes
     */
    bool thisJobWritesThisCalib(size_t calibNumber) const;

    /**
     *  @brief notifies SplitScanMgr of a new external link that will go in the master file.
     *
     *  Meant to by called by split mode master (job 0 not mpi mode). Adds link to write to master
     *  list.
     *
     *  @param[in] linkName     - the group name for the link.
     *  @param[in] calibNumber  - the calib cycle number that we link to
     *  @param[in] linkGroupLoc - the group where the link is made.
     *
     *  Typically linkGroupLoc will be for Run:0000 in the master file, and if calibNumber is,
     *  for example 3, linkName will be CalibCycle:0003. The filename linked to will be a
     *  relative path to a file in the same directory as the master file.
     */
    void newExtLnkForMaster(const char *linkName,
                              size_t calibCycle,
                              hdf5pp::Group & linkGroupLoc);

    /// enum for updateMasterLinks
    enum UpdateExtLinksMode {writeAll,            /// add all links to the master file
                             writeFinishedOnly};  /// only add links for finished files
    static std::string updateModeToStr(enum UpdateExtLinksMode mode); /// dump function for enum

    /**
     *  @brief updates the external links in the master file.
     *
     *  Takes an enum to indicate whether or not to only add links to finished files,
     *  or all remaining links. Only meant to be called for non mpi split scan master
     */
    void updateMasterLinks(enum UpdateExtLinksMode updateMode);

    /**
     *  @brief create the given calib cycle group in the calib file for the group.
     *
     * Meant to be called from both MPIWorker and Split mode.
     *
     *  @param[in]    calibCycle - calib cycle number
     *  @return hdf5pp::Group to CalibCycle:000x within the external file.
     */
    hdf5pp::Group createCalibCycleGroupInExtCalibFile(size_t calibCycle);

    /**
     *  @brief create a configure group in the external calib file.
     *
     *  Creates the group /config in the file. Throws if file has not been created.
     *
     *  Meant to be called from MPIWorker only.
     *
     *  @param[in]    calibCycle - calib cycle number
     *  @return hdf5pp::Group to /Config within the external file.
     */
    hdf5pp::Group createConfigureGroupInExtCalibFile(size_t calibCycle);

    /**
     * creates h5 file based on given calibCycle.
     */
    hdf5pp::File createCalibCycleFile(size_t calibCycle);

    /**
     * @brief creates calib file if needed.
     *
     * In MPI mode, there may be several calib cycles in the same file.
     * SplitScanMgr will check if the given calib cycle is part of a larger file,
     * and not create the file.
     *
     * In non MPI mode, with one calib cycle per file, this usually creates the file.
     *
     * @return true if created the file
     */
    bool createCalibFileIfNeeded(size_t calibCycle);

    /// returns the group for the root entry in the external hdf5 file where this 
    /// calib cycle is being written to
    hdf5pp::Group extCalibFileRootGroup(size_t calibCycle);

    /// close a calib cycle file corresponding to this calib cycle
    void closeCalibCycleFile(size_t calibCycle);

    /**
     * @brief makes an external link
     *
     * The filename linked to will be the 
     *
     * @param[in]     linkName - the group name of the link, and the group name of the target
     * @param[in]   calibCycle - which calib cycle this is, used to construct the target file name
     * @param[in] linkGroupLoc - the parent to to make the link in
     */
    bool createExtLink(const char *linkName,
		       const std::string &extH5File,
		       hdf5pp::Group & linkGroupLoc);

    /**
     * @brief returns the full target file path for the external calib cycle file.
     *
     * This path will have the same parent directory as the h5filePath argument that
     * SplitScanMgr was constructed with.
     * If SplitScanMgr was constructed with h5filePath = writeDir/output.h5 
     * then the filename returned will be writeDir/output_cc000x.h5 
     * (where x is the calibCycle)
     *
     * @param[in] calibCycle   - the calib cycle number
     * @return the filename, using the full path based on h5filePath
     */
    std::string getExtCalibCycleFilePath(size_t calibCycle);

    /// just base name
    std::string getExtCalibCycleFileBaseName(size_t calibCycle);

    /**
     * @brief returns true if the given calib cycle is finished.
     *
     * should only be called for non MPI split mode
     *
     * For example, if there are 3 jobs, and calibCycle is 2, 
     * checks to see if calibcyle 5 exists on disk - as that is
     * the next file that job 2 will make after finishing cc 2.
     *
     * If there is no 5th calib cycle to write, the routine will
     * return false even though cc2 may be done.
     *
     * @param[in] calibCycle   - the calib cycle number
     * @return true if file is ready
     */
    bool calibFileIsFinished(size_t calibCycle);


  private:
    std::string m_h5filePath;
    SplitMode  m_splitScanMode;
    int m_jobNumber;
    int m_jobTotal;
    int m_mpiWorkerStartCalibCycle;
    bool m_overwrite;
    int m_fileSchemaVersion;

    struct ExtCalib {
      hdf5pp::File file;
      hdf5pp::Group configGroup;
      std::map<size_t, hdf5pp::Group> groups;
      LusiTime::Time startTime;
    };
    // for splitMode - where the job writes many calib files, there
    // will be many entries in m_extCalib (below). For an MPIWorker, 
    // there will only be one - for the starting calib cycle
    std::map<size_t, ExtCalib > m_extCalib;

    /// helper function that deals with MPIWorker vs normal split scan mode
    size_t getExtCalibIndex(size_t calibCycle);

    /// gets ExtCalib for this cycle - must be present in map or exception
    ExtCalib & getExtCalib(size_t calibCycle);

    struct MasterLinkToWrite {
      std::string linkName;
      hdf5pp::Group linkGroupLoc;
      MasterLinkToWrite()  {};
      MasterLinkToWrite(const char *_linkName, hdf5pp::Group &_group) :
        linkName(_linkName), linkGroupLoc(_group) {}
    };

    std::map< size_t, MasterLinkToWrite> m_masterLnksToWrite;
  };

} // namespace
#endif  // TRANSLATOR_H5OUTPUT_H
