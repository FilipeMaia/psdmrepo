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

    enum SplitMode { NoSplit, MPIWorker, MPIMaster };
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
                 bool ccInSubDir,
                 SplitMode splitScanMode,
                 int mpiWorkerStartCalibCycle);

    /// return true if responsible for main h5 file
    bool thisJobWritesMainOutputFile() const;

    /// returns true if operating in split scan mode
    bool splitScanMode() const { return m_splitScanMode != NoSplit; }

    /// returns true if MPIWorker
    bool isMPIWorker() const { return m_splitScanMode == MPIWorker; }

    /// returns true if MPIMaster
    bool isMPIMaster() const { return m_splitScanMode == MPIMaster; }

    /// creates the calib cycle files subdirectory if it does not exist and is specified in the 
    /// ccInSubDir option. Best to only call this function once from the master process to avoid any
    /// possible race condition with creating a directory.
    void createCCSubDirIfNeeded() const;

    /**
     * @brief makes an external link
     *
     * The filename linked to will be the 
     *
     * @param[in]     linkName - the group name of the link
     * @param[in]   targetName - the group name of the target in the external file, starting at /
     * @param[in]   calibCycle - which calib cycle this is, used to construct the target file name
     * @param[in] linkGroupLoc - the parent group to make the link in
     */
    bool createExtLink(const char *linkName,
                       const char *targetName,
                       const std::string &extH5File,
                       hdf5pp::Group & linkGroupLoc);

    /**
     * @brief returns the full target file path for the external calib cycle file.
     *
     * This is the path that should be used to create the external calib cycle files.
     *
     * If SplitScanMgr was constructed with h5filePath = writeDir/output.h5 and mpiFirstCalibCycle=3
     * this will return writeDir/output_cc0003.h5 or writeDir/output_ccfiles/output_cc0003.h5 depending
     * on if ccInSubdir is true.
     *
     * @param[in]  calibNumber - the calib cycle number. Defaults to -1 which means to use the Mpi worker starting calib number.
     * 
     * @return the filename, using the full path based on h5filePath
     */
    std::string getExtFilePath(int calibNumber = -1) const;

    /// the relative link for the master file
    std::string getExtFileForLink(int calibNumber = -1) const;

  protected:

    /// just base name. if the starting calib cycle is 3, and h5filePath is a/b/myout.h5
    /// this returns myout_cc0003.h5
    std::string getExtFileBaseName(int calibNumber = -1) const;

    // return cc subdir basename, if h5filePath is mydir/myfile.h5 then this 
    // returns myfile_ccfiles
    std::string getCCSubDirBaseName() const;

  private:
    std::string m_h5filePath;
    bool m_ccInSubDir;
    SplitMode m_splitScanMode;
    int m_mpiWorkerStartCalibCycle;
  };

} // namespace
#endif  // TRANSLATOR_H5OUTPUT_H
