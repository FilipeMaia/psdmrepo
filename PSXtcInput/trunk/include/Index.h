#ifndef PSXTCINPUT_INDEX_H
#define PSXTCINPUT_INDEX_H

#include <boost/cstdint.hpp>
#include <vector>
#include <queue>

#include "psana/Index.h"
#include "psana/Configurable.h"
#include "PSXtcInput/DgramPieces.h"

namespace PSXtcInput {

/// @addtogroup PSXtcInput

/**
 *  @ingroup PSXtcInput
 *
 *  @brief Interface to allow XTC file random access.
 *
 *  @version $Id: Index.h 7696 2014-02-27 00:40:59Z cpo@SLAC.STANFORD.EDU $
 *
 *  @author Christopher O'Grady
 */

class IndexRun;
class RunMap;

class Index : public psana::Index, public psana::Configurable {
public:
  Index(const std::string& name, std::queue<DgramPieces>& queue);
  ~Index();
  int jump(psana::EventTime t);
  void setrun(int run);
  void allowCorruptEpics();
  unsigned nsteps();
  void end();
  void times(psana::Index::EventTimeIter& begin, psana::Index::EventTimeIter& end);
  void times(unsigned nstep, psana::Index::EventTimeIter& begin, psana::Index::EventTimeIter& end);
  const std::vector<unsigned>& runs();
private:
  std::vector<std::string> _fileNames;
  std::queue<DgramPieces>& _queue;
  IndexRun*                _idxrun;
  RunMap*                  _rmap;
  int                      _run;
  bool                     _allowCorruptEpics;
};

}

#endif // PSXTCINPUT_INDEX_H
