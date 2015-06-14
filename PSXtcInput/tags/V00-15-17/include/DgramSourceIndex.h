#ifndef PSXTCINPUT_DGRAMSOURCEINDEX_H
#define PSXTCINPUT_DGRAMSOURCEINDEX_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DgramSourceIndex.h 7696 2014-02-27 00:40:59Z cpo@SLAC.STANFORD.EDU $
//
// Description:
//	Class DgramSourceIndex.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <queue>

//----------------------
// Base Class Headers --
//----------------------
#include "PSXtcInput/IDatagramSource.h"
#include "PSXtcInput/DgramPieces.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/// @addtogroup PSXtcInput

/**
 *  @ingroup PSXtcInput
 *
 *  @brief Implementation of IDatagramSource interface which reads data from input files.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: DgramSourceIndex.h 7696 2014-02-27 00:40:59Z cpo@SLAC.STANFORD.EDU $
 *
 *  @author Christopher O'Grady
 */

class DgramSourceIndex : public IDatagramSource {
public:

  DgramSourceIndex();
  virtual ~DgramSourceIndex();
  virtual void init();
  void setQueue(std::queue<DgramPieces>& queue) {_queue = &queue;}
  virtual bool next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg);

private:
  std::queue<DgramPieces>* _queue;
};

} // namespace PSXtcInput

#endif // PSXTCINPUT_DGRAMSOURCEINDEX_H
