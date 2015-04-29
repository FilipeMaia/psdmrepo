#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

#include <vector>

#include "psana_test/xtciter.h"

#include "pdsdata/xtc/Xtc.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"
#include "psana_test/XtcIterator.h"

using namespace Pds;
using namespace psana_test;

namespace psana_test {
// ----------------------------------------------
// DgramHeaderIteratorImpl
class DgramHeaderIteratorImpl {
public:
  DgramHeaderIteratorImpl(int fd) :  _fd(fd), 
                                     _posNextRead(0) {};

  Pds::Dgram * next() { 
    if (::read(_fd,&_dg,sizeof(_dg))==0) return 0;
    ::lseek(_fd,_dg.xtc.sizeofPayload(),SEEK_CUR);
    _posNextRead += sizeof(Pds::Dgram);
    _posNextRead += _dg.xtc.sizeofPayload();
    return &_dg;
  }
  size_t posNextDgram() { return _posNextRead; }
private:
  int _fd;
  Pds::Dgram _dg;
  size_t _posNextRead;
};

// ----------------------------------------------
// DgramHeaderIterator 

DgramHeaderIterator::DgramHeaderIterator(int fd) {
  _impl = new DgramHeaderIteratorImpl(fd);
};

DgramHeaderIterator::~DgramHeaderIterator() {
  delete _impl;
};

Pds::Dgram * DgramHeaderIterator::next() {
  return _impl->next();
}

std::pair<Pds::Dgram *,size_t> DgramHeaderIterator::nextAndOffsetFromStart() {
  size_t offsetNext = _impl->posNextDgram();
  return std::make_pair<Pds::Dgram *,size_t>(_impl->next(), offsetNext);
}


// ----------------------------------------------
// DgramWithXtcPayloadIterator 

DgramWithXtcPayloadIterator::DgramWithXtcPayloadIterator(int fd,size_t maxDgramSize, bool xtcDiag) : 
  _posNextDgram(0)
{
  // TODO: presently we don't do diagnostics on the dgram iteration, add it based on xtcDiag flag?
  _impl = new Pds::XtcFileIterator(fd,maxDgramSize); 
};

DgramWithXtcPayloadIterator::~DgramWithXtcPayloadIterator() {
  delete _impl;
};

Pds::Dgram * DgramWithXtcPayloadIterator::next() {
  Pds::Dgram *dg =  _impl->next();
  if (dg) {
    _posNextDgram += sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
  }
  return dg;
}

std::pair<Pds::Dgram *,size_t> DgramWithXtcPayloadIterator::nextAndOffsetFromStart() {
  Pds::Dgram *dg =  _impl->next();
  size_t dgOffset = _posNextDgram;
  if (dg) {
    _posNextDgram += sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
  }
  return std::pair<Pds::Dgram *, size_t>(dg,dgOffset);
}


//----------------------------------------------------
class ListXtcIterator : public psana_test::XtcIterator {
public:
  ListXtcIterator(Pds::Xtc *root, 
                  std::vector<XtcDepthOffset> & list, 
                  int depth, 
                  uint8_t * baseOffset,
                  bool xtcDiagnose) : 
    psana_test::XtcIterator(root, xtcDiagnose), 
    _list(list), 
    _depth(depth), 
    _baseOffset(baseOffset),
    _xtcDiagnose(xtcDiagnose) {}

  virtual int process(Pds::Xtc * xtc) {
    size_t xtcOffset = (size_t)(((uint8_t*)xtc) - _baseOffset);
    _list.push_back(XtcDepthOffset::makeXtcDepthOffset(xtc,_depth,xtcOffset));
    if (xtc->contains.id() == Pds::TypeId::Id_Xtc) {
      ListXtcIterator iterChildren(xtc, _list, _depth+1, _baseOffset, _xtcDiagnose);
      iterChildren.iterate();
    }
    return 1; // keep iterating
  }
private:
  std::vector<XtcDepthOffset> & _list;
  int _depth;
  uint8_t *_baseOffset;
  bool _xtcDiagnose;
};

class XtcChildrenIteratorImpl {
  // uses the psana_test::XtcIterator (based on Pds::XtcIterator) to 
  // recursively to through the xtc container
  // and adds xtcs to a list.  The user then goes through this list.
  // inefficient as we process the xtc container twice - but reuses code
  // coming from pds library to iterate over an xtc
public:
  XtcChildrenIteratorImpl(Pds::Xtc *root, int startingDepth, bool xtcDiagnose) : _pos(0) {
    ListXtcIterator * _listXtcIter = new ListXtcIterator(root,
                                                         _childrenXtcAndPositions,
                                                         startingDepth,
                                                         (uint8_t *)root,
                                                         xtcDiagnose);
    _listXtcIter->iterate();
    delete _listXtcIter;
  }
  ~XtcChildrenIteratorImpl() {}

  XtcDepthOffset nextWithPos() { 
    if (_pos >= _childrenXtcAndPositions.size()) {
      return XtcDepthOffset::makeXtcDepthOffset(0,0,0);
    }
    XtcDepthOffset nextXtc = _childrenXtcAndPositions.at(_pos);
    _pos += 1;
    return nextXtc;
  }
  Pds::Xtc * next() { 
    XtcDepthOffset nextXtc = nextWithPos();
    return nextXtc.xtc;
  }
private:
  std::vector<XtcDepthOffset> _childrenXtcAndPositions;
  unsigned  _pos;
};

XtcChildrenIterator::XtcChildrenIterator(Pds::Xtc *root, int startingDepth, bool xtcDiagnose) {
  _impl = new XtcChildrenIteratorImpl(root,startingDepth, xtcDiagnose);
}
XtcChildrenIterator::~XtcChildrenIterator() {
  delete _impl;
}

Pds::Xtc * XtcChildrenIterator::next() { return _impl->next(); }

  XtcDepthOffset XtcChildrenIterator::nextWithPos() { return _impl->nextWithPos(); }

//------------------------------------------------------------
// 
DgramPosIterator::DgramPosIterator(int fd, size_t maxDgramSize) :
  _fd(fd),
  _maxDgramSize(maxDgramSize),
  _buf(new char[maxDgramSize]),
  _nextDgramOffsetInFile(0)
{}

DgramPosIterator::~DgramPosIterator() {
  delete[] _buf;
}

void DgramPosIterator::next(Dgram * & dgram, int64_t & filePos) {
  Dgram& dg = *(Dgram*)_buf;
  filePos = _nextDgramOffsetInFile;
  dgram = &dg;
  if (::read(_fd, &dg, sizeof(dg))==0) {
    dgram = 0;
    return;
  }
  size_t payloadSize = dg.xtc.sizeofPayload();
  if ((payloadSize+sizeof(dg))>_maxDgramSize) {
    fprintf(stderr,"Error: Datagram size %zu larger than maximum: %zu\n", payloadSize+sizeof(dg), _maxDgramSize);
    dgram = 0;
    return;
  }
  ssize_t sz = ::read(_fd, dg.xtc.payload(), payloadSize);
  if (sz != (ssize_t)payloadSize) {
    dgram = 0;
    fprintf(stderr,"DgramPosIterator::next read incomplete payload %d/%d\n",
     (int) sz, (int) payloadSize);
    return;
  }
  _nextDgramOffsetInFile += sizeof(Dgram) + payloadSize;
}

/* ------------------------------------------------
  class XtcOffsetIterator
  ----------------------------- */
void XtcOffsetIterator::iterate(Pds::Xtc* root, int depth) {
  if (root->damage.value() & ( 1 << Pds::Damage::IncompleteContribution)) {
    fprintf(stderr,"root damage = %X which has bit %X on (incomplete contrib), returning\n",
           root->damage.value(), Pds::Damage::IncompleteContribution);
    return;
  }
  Pds::Xtc* xtc     = (Pds::Xtc*)root->payload();
  int remaining = root->sizeofPayload();
  
  while(remaining > 0)  {
    if(xtc->extent==0) {
      fprintf(stderr, "trying to skip corrupt event. xtc->extent is 0, but remaining is > 0\n");
      break;
    }
    int32_t xtcOffset = int32_t((uint8_t *)xtc-m_offsetFrom);
    if(!process(xtc,xtcOffset,depth)) break;
    if (xtc->contains.id()==Pds::TypeId::Id_Xtc) iterate(xtc,depth+1);
    remaining -= xtc->sizeofPayload() + sizeof(Pds::Xtc);
    xtc      = xtc->next();
  }
  return;
}


} // namespace psana_test
