#ifndef XTCFILES_POSITION_H
#define XTCFILES_POSITION_H

//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcFilesPosition
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <vector>
#include <string>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/XtcFileName.h"
#include "PSEvt/Event.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//             ---------------------
//             -- Class Interface --
//             ---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief class provides position for event - filenames and offsets
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author David Schneider
 */

class XtcFilesPosition {
 public:
  XtcFilesPosition() {}

  /**
   * @brief constructor takes lists of filenames and offsets.
   *
   * Input parameters are of type list so as to work with the configList 
   * function for getting config parameters
   *
   * @param[in] fileNames list of filenames
   * @param[in] offsets list of offsets
   *
   * @throw ArgumentException if any of the following occur: 
   *    the number of fileNames is not equal to the number of offsets,
   *    the same stream appears more than once in the files list,
   *    there is more than one run number among the filenames
   */
  XtcFilesPosition(const std::list<std::string> &fileNames, 
		   const std::list<off64_t> &offsets);

  /**
   * @brief utitilty function to produce XtcFilesPosition from event.
   *
   * Relies on DgramList being present in the event. If not,
   * returns a null pointer. If so, constructs XtcFilesPosition
   * using files and offsets in the DgramList.
   *
   * @return Shared pointer to XtcFilesPosition
   *
   * @throw ArgumentException as per rules in XtcFilesPosition constructor
   */
  static boost::shared_ptr<XtcFilesPosition> makeSharedPtrFromEvent(PSEvt::Event &evt);

  /// the run for the files in this position
  int run() const { return m_run; }

  /// returns true if a there is a file in the position from the given stream
  bool hasStream(int stream) const { return m_streamToPos.find(stream) != m_streamToPos.end(); }

  /**
   * @brief returns the filename and offset for a given stream.
   *
   * @throw ArgumentError if this stream is not part of the files in the position
   */
  std::pair<XtcFileName,off64_t> getChunkFileOffset(int stream) const;

  size_t size() const { return m_offsets.size(); }

  std::vector<off64_t> offsets() const { return m_offsets; }

  std::vector<std::string> fileNames() const;
  
 private:
  std::vector<XtcInput::XtcFileName> m_xtcFileNames;
  std::vector<off64_t> m_offsets;
  int m_run;
  std::map<int,std::pair<XtcFileName,off64_t> > m_streamToPos;
};

}; // namespace XtcInput

#endif
