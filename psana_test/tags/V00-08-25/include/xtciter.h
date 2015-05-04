#ifndef PSANA_TEST_FILE_XTCITER_H
#define PSANA_TEST_FILE_XTCITER_H

#include <stdint.h>
#include <stddef.h>  // size_t

#include <utility>  // std::pair

namespace Pds {
  class Dgram;
  class Xtc;
  class XtcFileIterator;
}

namespace psana_test {
  
  class DgramHeaderIteratorImpl;
  
  // Iterates through the Dgram headers of a previously opened xtc file.
  // Does not read in the xtc payloads.
  class DgramHeaderIterator {
  public:
    DgramHeaderIterator(int fd); // construct with open file handle
    ~DgramHeaderIterator(); 
    Pds::Dgram *next();             
    std::pair<Pds::Dgram *, size_t> nextAndOffsetFromStart();
  private:
    DgramHeaderIteratorImpl *_impl;
  };
  
  
  // Iterates through the Dgrams of a previously opened xtc file.
  // Reads in the Xtc Payload for each Dgram.  
  // Since the payload is read in, the Dgram Xtc can be used to 
  // iterate the Xtc's.
  class DgramWithXtcPayloadIterator {
  public:
    DgramWithXtcPayloadIterator(int fd, size_t maxDgramSize=1<<26); // construct with open file handle
    ~DgramWithXtcPayloadIterator(); 
    Pds::Dgram *next();             
    std::pair<Pds::Dgram *, size_t> nextAndOffsetFromStart();
  private:
    Pds::XtcFileIterator *_impl;
    int64_t _posNextDgram;
  };
  
  // helper class to return depth of of xtc in container, and offset from root
  struct XtcDepthOffset {
    Pds::Xtc *xtc;
    int depth;
    size_t offset;
    static XtcDepthOffset  makeXtcDepthOffset(Pds::Xtc *xtc, int depth, size_t offset) { 
      XtcDepthOffset xdo;
      xdo.xtc = xtc;
      xdo.depth = depth;
      xdo.offset = offset;
      return xdo;
    };
  };

  class XtcChildrenIteratorImpl;

  class XtcChildrenIterator {
  public:
    
    XtcChildrenIterator(Pds::Xtc * root, int startingDepth);
    ~XtcChildrenIterator();
    Pds::Xtc * next();
    XtcDepthOffset nextWithPos();
  private:
    XtcChildrenIteratorImpl * _impl;
  };
  
// this class started as a copy of pdsdata/xtc/XtcIterator and was
// then modified to return the dgram offset in the file.
class DgramPosIterator {
public:
  DgramPosIterator(int fd, size_t maxDgramSize);
  ~DgramPosIterator();
  void next(Pds::Dgram * & dgram, int64_t & filePos);
private:
  int      _fd;
  size_t   _maxDgramSize;
  char*    _buf;
  int64_t  _nextDgramOffsetInFile;
};


/* started with a copy of code for pdsdata/xtc/XtcIterator
   modified to also return the offset of the xtc
   from the starting pointer. Keeps track of xtc depth.  
   Now Will recurse into other xtc.  So if an xtc is Id_Xtc, 
   we recursively work through it.
 */ 
class XtcOffsetIterator  {
public:
 XtcOffsetIterator(Pds::Xtc* root, 
                   uint8_t *offsetFrom, 
                   int depth) 
   : m_root(root), 
    m_depth(depth), 
    m_offsetFrom(offsetFrom) {};

  virtual ~XtcOffsetIterator() {}
  virtual bool process(Pds::Xtc* xtc, int offset,int depth) = 0;
  void iterate() {iterate(m_root,m_depth); };
  void iterate(Pds::Xtc*,int);
  const Pds::Xtc* root() const { return m_root; };
  Pds::Xtc* m_root;
  int m_depth;
  uint8_t *m_offsetFrom;
};

} // namespace psana_test

#endif // #define PSANA_TEST_FILE_XTCITER_H
