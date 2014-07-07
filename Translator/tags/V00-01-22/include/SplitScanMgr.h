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
   * In split scan mode, Each calib cycle is written into a separate file by potentially different
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
   * newCalibCycleExtLink - This should be called by job0. job0 is soley responsible for manipulating
   *                        the master file with links to the calib cycle files. This routine does not
   *                        neccessarily add the link to the master file when called. Ideally we only add 
   *                        a link after the calib cycle file is finished (so as to eliminate errors 
   *                        that programs will get get by following links to unfinished, or non existant files).
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
    /**
     *  @brief constructor for SplitScanMgr
     *
     *  @param[in] h5filePath   - full path name for the master file.
     *                            All calib cycle files will be created in the same directory.
     *  @param[in] splitScanMode - true if this is split scan mode, if False SplitScanMgr does little.
     *  @param[in] linkGroupLoc - the group where the link is made.
     *
     *  Typically linkGroupLoc will be for Run:0000 in the master file, and if calibNumber is
     *  for example 3, linkName will be CalibCycle:0003. If the master filename is output.h5,
     *  the filename linked to will be output_cc0003.h5. It is always a relative path in the 
     *  current directory.
     */
    SplitScanMgr(const std::string &h5filePath, bool splitScanMode,
              int jobNumber, int jobTotal, bool overwrite,
              int fileSchemaVersion);

    /// returns true if operating in split scan mode
    bool splitScanMode() const { return m_splitScanMode; }

    /// this job number (0 if not split scan mode)
    int jobNumber() const { return m_jobNumber; }

    /// total number of jobs (1 if not split scan mode)
    int jobTotal() const { return m_jobTotal; }

    /// true if not in split scan mode, or split scan mode and this is job 0
    bool noSplitOrJob0() const;

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
     *  Presently the link is immediately created in the master file, no check is made to 
     *  make sure the link file exists and is finished.
     *
     *  @param[in] linkName     - the group name for the link.
     *  @param[in] calibNumber  - the calib cycle number that we link to
     *  @param[in] linkGroupLoc - the group where the link is made.
     *
     *  Typically linkGroupLoc will be for Run:0000 in the master file, and if calibNumber is,
     *  for example 3, linkName will be CalibCycle:0003. The filename linked to will be a
     *  relative path to a file in the same directory as the master file.
     */
    void newCalibCycleExtLink(const char *linkName,
                              size_t calibCycle,
                              hdf5pp::Group & linkGroupLoc);

    /// enum for updateCalibCycleExtLinks
    enum UpdateExtLinksMode {writeAll,            /// add all links to the master file
                             writeFinishedOnly};  /// only add links for finished files

    /**
     *  @brief updates the external links in the master file.
     *
     *  Takes an enum to indicate wether or not to only add links to finished files,
     *  or all remaining links. Presently not implemented.
     */
    void updateCalibCycleExtLinks(enum UpdateExtLinksMode updateMode);

    /**
     *  @brief create the next external calib cycle file and returns a group within the file.
     *
     *  Creates a file in the same directory as the master filename that SplitScanMgr was
     *  constructed with. The filename is given by getExtCalibCycleFilePath. 
     *  Creates one group in the file: CalibCycle:000x where x is calibCycle.
     *  Returns the group.
     *
     *  @param[in]    calibCycle - calib cycle number
     *  @return hdf5pp::Group to CalibCycle:000x within the external file.
     */
    hdf5pp::Group createNextCalibCycleFile(size_t calibCycle);

    /// close a calib cycle file.
    void closeCalibCycleFile(size_t calibCycle);

    /**
     * @brief makes an external link to a calib file.
     *
     * The filename linked to will be the 
     *
     * @param[in]     linkName - the group name of the link, and the group name of the target
     * @param[in]   calibCycle - which calib cycle this is, used to construct the target file name
     * @param[in] linkGroupLoc - the parent to to make the link in
     */
    bool createCalibCycleExtLink(const char *linkName,
                                 size_t calibCycle,
                                 hdf5pp::Group & linkGroupLoc);

    /**
     * @brief returns the full target file path to use for the external calib cycle file.
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

  private:
    std::string m_h5filePath;
    bool m_splitScanMode;
    int m_jobNumber;
    int m_jobTotal;
    bool m_overwrite;
    int m_fileSchemaVersion;

    struct ExtCalib {
      bool linkAddedToMaster;
      hdf5pp::File file;
      hdf5pp::Group group;
      size_t calibNumber;
      LusiTime::Time startTime;
    };
    std::map< size_t, ExtCalib > m_extCalib;
  };

} // namespace

#endif  // TRANSLATOR_H5OUTPUT_H
