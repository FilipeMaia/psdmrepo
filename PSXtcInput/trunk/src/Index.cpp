//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: Index.cpp 7696 2014-02-27 00:40:59Z cpo@SLAC.STANFORD.EDU $
//
// Description:
//	Class Index...
//
// Author List:
//      Christopher O'Grady
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/Index.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <string>
#include <iomanip>
#include <fcntl.h>
#include <stdlib.h>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSXtcInput/Exceptions.h"
#include "pdsdata/index/IndexFileStruct.hh"
#include "pdsdata/index/IndexFileReader.hh"
#include "pdsdata/index/IndexList.hh"
#include "IData/Dataset.h"
#include "XtcInput/XtcFileName.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;
using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace {
  const char* logger = "PSXtcInput::Index";
}

namespace PSXtcInput {

// class to take a list of xtc filenames and generate a map
// where one can look up the files for a particular run.

class RunMap {
public:
  std::vector<XtcInput::XtcFileName> files;
  typedef std::map<unsigned, std::vector<XtcInput::XtcFileName> > map;
  std::vector<unsigned> runs;
  map runFiles;

  RunMap(std::vector<std::string> &m_fileNames) {
    // Input can be a mixture of files and datasets.
    // Live mode is not supported. "one-stream mode"
    // is only supported if the users provides a list of
    // timestamps from one stream.

    typedef std::vector<std::string> FileList;

    // guess whether we have datasets or pure file names (or mixture)
    for (FileList::const_iterator it = m_fileNames.begin(); it != m_fileNames.end(); ++ it) {
    
      IData::Dataset ds(*it);
      if (ds.exists("live")) MsgLog(logger, fatal, "Live mode not supported with xtc indexing");
    
      if (ds.isFile()) {

        // must be file name
        files.push_back(XtcInput::XtcFileName(*it));
        
      } else {

        // Find files on disk and add to the list
        const IData::Dataset::NameList& strfiles = ds.files();
        if (strfiles.empty()) MsgLog(logger, fatal, "Empty file list");
        for (IData::Dataset::NameList::const_iterator it = strfiles.begin(); it != strfiles.end(); ++ it) {
          XtcInput::XtcFileName file(*it);
          files.push_back(file);
        }
      }
      // sort files to make sure we get a chunk0 first
      sort(files.begin(),files.end());
      
      // sort all files according run
      for (std::vector<XtcInput::XtcFileName>::const_iterator it = files.begin(); it != files.end(); ++ it) {
        runFiles[it->run()].push_back(*it);
      }
      for (map::const_iterator it = runFiles.begin(); it != runFiles.end(); ++ it) {
        runs.push_back(it->first);
      }
    }
  }
};

// class which manages xtc files, including "jump" function to do random access

class IndexXtcReader {
public:
  IndexXtcReader(const vector<XtcFileName> &xtclist) {
    _nfiles = xtclist.size();
    _fd.resize(_nfiles);

    for (vector<string>::size_type ifile=0; ifile!=_nfiles; ifile++) {
      _fd[ifile] = ::open(xtclist[ifile].path().c_str(), O_RDONLY | O_LARGEFILE);
      if (_fd[ifile]==-1) MsgLog(logger, fatal,
                                 "File " << xtclist[ifile].path().c_str() << " not found");
    }
  }

  ~IndexXtcReader() {
    for (unsigned i=0;i<_nfiles;i++) ::close(_fd[i]);
  }

  Pds::Dgram* jump(int file, int64_t offset) {
    int64_t found = lseek64(_fd[file],offset, SEEK_SET);
    if (found != offset) {
      stringstream ss;
      ss << "Jump to offset " << offset << " failed";
      MsgLog(logger, error, ss.str());
      throw IndexSeekFailed(ERR_LOC);
    }
    Pds::Dgram* dg = (Pds::Dgram*)new char[MaxDgramSize];
    if (::read(_fd[file], dg, sizeof(*dg))==0) {
      return 0;
    } else {
      ::read(_fd[file], dg->xtc.payload(), dg->xtc.sizeofPayload());
      return dg;
    }
  }

private:
  enum {MaxDgramSize=0x800000};
  unsigned _nfiles;
  vector<int> _fd;
};

// class which is used by IndexBase. for each event in the index table,
// keeps track of which xtc-file-number contains the associated epics
// data, as well as the offset to that data (used by "jump").

class EpicsInfo {
public:  
  EpicsInfo() : offset(-1),file(-1) {}
  int64_t offset;
  int file;
  bool operator==(const EpicsInfo& other) const {return (offset==other.offset && file==other.file);}
  bool operator!=(const EpicsInfo& other) const {return !(*this==other);}
};

// this class is one entry in a index table.  it is used mainly for L1Accepts,
// but is also reused for the table of BeginCalibs (hence the template)

template<typename T>
class IndexBase {
public:
  T entry;
  int file;
  void _init() {
    file=-1;
  }
  virtual ~IndexBase() {}
  IndexBase() {}
  IndexBase(uint32_t seconds, uint32_t nanoseconds) {
    entry.uSeconds=seconds;
    entry.uNanoseconds=nanoseconds;
    _init();
  }
  bool operator<(const IndexBase& other) const {
    return entry.time()<other.entry.time();
  }
  bool operator==(const IndexBase& other) const {
    return entry.time()==other.entry.time();
  }
  bool operator!=(const IndexBase& other) const {
    return !(*this==other);
  }
};

typedef IndexBase<Pds::Index::CalibNode> IndexCalib;
typedef IndexBase<Pds::Index::L1AcceptNode> IndexUnixTime;

// used by the IOC recorders when we want to sort/search
// using unix timestamp, but not the fiducial

class IndexFiducial : public IndexUnixTime {
public:
  IndexFiducial() {}
  IndexFiducial(uint32_t seconds, uint32_t nanoseconds, uint32_t fiducial) : IndexUnixTime(seconds,nanoseconds) {
    entry.uFiducial=fiducial;
  }
};

class IndexEvent : public IndexFiducial {
public:
  enum {MaxEpicsSources=4};
  enum {SecondsLimit=5};
  // the code would be neater if this was a vector, but I think it would
  // be less performant as well, hence the hardwired number. - cpo
  EpicsInfo einfo[MaxEpicsSources];
  virtual ~IndexEvent() {}
  IndexEvent() {}
  IndexEvent(uint32_t seconds, uint32_t nanoseconds, uint32_t fiducial) : IndexFiducial(seconds,nanoseconds,fiducial) {}
  bool operator<(const IndexEvent& other) const {
    //    return entry.time()<other.entry.time();
    int64_t t1sec = entry.uSeconds;
    int64_t t2sec = other.entry.uSeconds;
    // if the timestamp has changed by "a lot" use that to decide event ordering
    if (abs(t1sec-t2sec)>SecondsLimit) return t1sec<t2sec;
    // if the timestamp has changed by "a little" use the fiducial to decide event ordering
    // shift the 17-bit fiducial value over so we can use signed-arithmetic
    int32_t f1 = entry.uFiducial<<15;
    int32_t f2 = other.entry.uFiducial<<15;
    // do a sanity check: include a factor of 2 "headroom"
    const int32_t maxdiff = (SecondsLimit*360*2)<<15;
    if (abs(f2-f1)>maxdiff) MsgLog(logger, fatal, "Fiducial sanity check failed.  fiducial1 " << entry.uFiducial << " fiducial2" << other.entry.uFiducial);
    return f1<f2;
  }
  bool operator==(const IndexEvent& other) const {
    return entry.time()==other.entry.time() && entry.uFiducial==other.entry.uFiducial;
  }
  bool operator!=(const IndexEvent& other) const {
    return !(*this==other);
  }
};

ostream& operator<<(ostream& os, const IndexEvent& idx) {
  os << "time " << std::hex << idx.entry.uSeconds << "/" << idx.entry.uNanoseconds << " fiducial " << idx.entry.uFiducial << ", filenum " << idx.file;
    return os;
}

// this is the implementation of the per-run indexing.  shouldn't be too
// hard to make it work for for per-calibcycle indexing as well.

class IndexRun {
private:
  // read an index file corresponding to an xtc file
  bool _getidx(const XtcFileName &xtcfile, Pds::Index::IndexList& idxlist) {
    string idxname = xtcfile.path();
    string basename = xtcfile.basename();
    size_t pos = idxname.find(basename,0);
    idxname.insert(pos,"index/");
    idxname.append(".idx");
    int fd = open(idxname.c_str(), O_RDONLY | O_LARGEFILE);
    if (fd < 0) {
      MsgLog(logger, warning, "Unable to open xtc index file " << idxname);
      return 1;
    }
    idxlist.readFromFile(fd);
    ::close(fd);
    return 0;
  }

  // this is tricky.  the bit-list of DAQ "detector sources" can be different
  // for the various event-nodes ("streams").  returns a bitmask of the bits
  // in the DAQ index TSegmentToIdMap that correspond to Epics types, as
  // well as a map (bit2Src) from those bits to the Src.  This latter map essentially
  // "inverts" the direction of TSegmentToIdMap.
  unsigned _getEpicsBit2SrcMap(const Pds::Index::IndexList::TSegmentToIdMap& seg, std::map<int,Pds::Src> &bit2Src) {
    unsigned mask = 0;
    for (Pds::Index::IndexList::TSegmentToIdMap::const_iterator it=seg.begin(); it!=seg.end(); ++it) {
      const Pds::Index::L1SegmentId::TTypeList& type = it->second.typeList;
      const Pds::Index::L1SegmentId::TSrcList& src = it->second.srcList;
      Pds::Index::L1SegmentId::TSrcList::const_iterator itsrc=src.begin();
      for (Pds::Index::L1SegmentId::TTypeList::const_iterator ittype=type.begin(); ittype!=type.end(); ++ittype) {
        Pds::TypeId::Type type = (*ittype).id();
        int index = it->second.iIndex;
        Pds::Src src = *itsrc;
        if (type==Pds::TypeId::Id_Epics) {
          bit2Src[1<<index]=src;
          mask |= 1<<index;
        }
        ++itsrc;
      }
    }
    return mask;
  }

  // add a single DAQ index-file to the large IndexBase table (either IndexEvent
  // or IndexCalib)
  template <typename T1, typename T2>
  void _store(T1 &idx, const T2 &add, vector<string>::size_type ifile) {
    int numadd = add.size();
    int numtot = idx.size();
    idx.resize(numadd+numtot);
    for (int i=0; i<numadd; i++) {
      idx[numtot].entry=add[i];
      idx[numtot].file = ifile;
      numtot++;
    }
  }

  // create a vector of unique times that the user can use to jump to events
  void _fillTimes() {
    IndexEvent last(0,0,0);
    for (vector<IndexEvent>::iterator itev = _idx.begin(); itev != _idx.end(); ++ itev) {
      if (*itev!=last) {
        _times.push_back(psana::EventTime(itev->entry.time(),itev->entry.uFiducial));
        last = *itev;
      }
    }
  }

  // add a datagram with "event" data (versus nonEvent data, like epics)
  // to the vector of pieces (i.e. add another "piece")
  void _add(Pds::Dgram* dg) {
    _pieces.eventDg.push_back(XtcInput::Dgram(XtcInput::Dgram::make_ptr(dg),XtcFileName("")));
  }

  // copy the event-pieces onto the queue where the DgramSourceIndex object
  // can pick them up.
  void _post() {
    _queue.push(_pieces);
  }

  // add only one "event" datagram and post
  void _post(Pds::Dgram* dg) {
    _add(dg);
    _post();
  }

  // post only this dg
  void _postOneDg(Pds::Dgram* dg) {
    _pieces.reset();
    if (dg) _post(dg);
  }

  // look for configure in first 2 datagrams from the first file.  this will fail
  // if we don't get a chunk0 first in the list of files.  we have previously
  // sorted the files in RunMap to ensure this is the case.
  void _configure() {
    int64_t offset = 0;
    for (int i=0; i<2; i++) {
      Pds::Dgram* dg = _xtc.jump(0, offset);
      if (dg->seq.service()==Pds::TransitionId::Configure) {
        _postOneDg(dg);
        _beginrunOffset = dg->xtc.sizeofPayload()+sizeof(Pds::Dgram);
        return;
      }
      offset+=dg->xtc.sizeofPayload()+sizeof(Pds::Dgram);
    }
    MsgLog(logger, fatal, "Configure transition not found in first 2 datagrams");
  }

  // send beginrun from the first file
  void _beginrun() {
    Pds::Dgram* dg = _xtc.jump(0, _beginrunOffset);
    if (dg->seq.service()!=Pds::TransitionId::BeginRun)
      MsgLog(logger, fatal, "BeginRun transition not found after configure transition");
    _postOneDg(dg);
  }

  // check to see if we need to send a begincalib by looking
  // to see if the begincalib needed by this timestamp
  // is the same as the begincalib we sent previously
  void _maybePostCalib(uint32_t seconds, uint32_t nanoseconds) {
    IndexCalib request(seconds,nanoseconds);
    vector<IndexCalib>::iterator it;
    it = lower_bound(_idxcalib.begin(),_idxcalib.end(),request);
    if (it==_idxcalib.begin()) {
      MsgLog(logger, fatal, "Calib cycle for time " << seconds << "/" << nanoseconds << " not found");
    } else {
      vector<IndexCalib>::iterator calib;
      calib=it-1;
      if (*calib!=_lastcalib) {
        // it appears that psana takes care of sending endcalib for us
        // need to send begincalib
        _postOneDg(_xtc.jump((*calib).file, (*calib).entry.i64Offset));
        _lastcalib=*calib;
      }
    }
  }

  // look through the ioc index table to find datagrams
  // within a time window of the DAQ dg, where the fiducials
  // match precisely
  void _maybeAddIoc(uint32_t seconds, uint32_t fiducial) {
    const unsigned window = 5; // in seconds
    vector<IndexFiducial>::iterator it;
    it = lower_bound(_idxioc.begin(),_idxioc.end(),IndexFiducial(seconds-window,0,0));
    while (it!=_idxioc.end()) {
      if (it->entry.uSeconds>(seconds+window)) return; // out of the window
      if (it->entry.uFiducial==fiducial) {
        Pds::Dgram* dg=0;
        dg = _xtc.jump((*it).file, (*it).entry.i64OffsetXtc);
        _add(dg); // don't return: there can be a match from another ioc stream
      }
      it++;
    }
  }

  // loop over the _lastEpics list (one per epics source).
  // look in the big IndexEvent table and see if this L1 timestamp needs
  // a new "nonEvent" datagram of epics info.
  void _maybeAddEpics(const IndexEvent& evt, const vector<Pds::Src>& src, Pds::Dgram* dg) {
    int i=0;
    for (vector<EpicsInfo>::iterator last =_lastEpics.begin(); last!=_lastEpics.end(); last++) {
      EpicsInfo request = evt.einfo[i];
      // check if we need new epics info
      if (*last != request) {
        // don't send if already have the epics data inside this event
        if (request.file != evt.file || request.offset != evt.entry.i64OffsetXtc) {
          Pds::Dgram* epicsdg = _xtc.jump(request.file,request.offset);
          if (epicsdg) {
            _pieces.nonEventDg.push_back(XtcInput::Dgram(XtcInput::Dgram::make_ptr(epicsdg),XtcFileName("")));
          } else {
            MsgLog(logger, fatal, "Epics data not found at offset" << request.offset);
          }
          *last=request;
        }
      }
      i++;
    }
    // if there are multiple EPICS sources, sort them so that
    // the oldest is first in the list, so the newest data "wins"
    // when andy processes the non-event datagrams.
    sort(_pieces.nonEventDg.begin(),_pieces.nonEventDg.end());
  }

  typedef std::map<int,Pds::Src> epicsmap;

  // loop over XTC files, and store the corresponding DAQ index information
  // in the big IndexEvent table.  also take care of the tricky mapping
  // from epics "bitmask" to Src for each index file.
  void _storeIndex(const vector<XtcFileName> &xtclist, std::map<Pds::Src,int>& src2EpicsArray,
                   vector<epicsmap>& bit2SrcVec, vector<unsigned>& epicsmask) {
    bool ifirst = 1;
    for (vector<string>::size_type ifile=0; ifile!=xtclist.size(); ifile++) {
      Pds::Index::IndexList idxlist;
      // get the DAQ index file, if it exists
      if (_getidx(xtclist[ifile], idxlist)) continue;
      if (xtclist[ifile].stream()<80) {
        // store them in event table that includes DAQ data
        _store(_idx,idxlist.getL1(),ifile);
        // begincalibs are a little tricky, I believe.  in principle
        // a begincalib for an event could be in a previous chunk
        // so I think we need to put them in one big list for the
        // whole run and search (although we could also add them
        // to the one big IndexEvent table, like we do for epics data -cpo
        _store(_idxcalib,idxlist.getCalib(),ifile);

        // epics is also tricky, because the ordering of the different
        // sources can change in the different DAQ index files.
        // store which array-offset we are using for this epics source.
        // also store the Pds::Src values in the same order, which
        // we will use to go lookup the epics data to attach to the requested event
        epicsmap bit2Src;
        epicsmask.push_back(_getEpicsBit2SrcMap(idxlist.getSeg(),bit2Src));
        if (ifirst) {
          ifirst = 0;
          int i=0;
          for (epicsmap::const_iterator it=bit2Src.begin(); it!=bit2Src.end(); ++it) {
            src2EpicsArray[it->second]=i;
            i++;
            _epicsSource.push_back(it->second);
          }
        }
        bit2SrcVec.push_back(bit2Src);
      } else {
        // store them in event table that includes ioc data
        _store(_idxioc,idxlist.getL1(),ifile);
      }
    }
    _lastEpics.resize(_epicsSource.size());

    sort(_idx.begin(),_idx.end());
    sort(_idxcalib.begin(),_idxcalib.end());
    sort(_idxioc.begin(),_idxioc.end());
  }

  // go through the big IndexEvent table, and store nonEvent
  // datagram offsets to use for each epics source.
  void _updateEpics(std::map<Pds::Src,int>& src2EpicsArray,
                    vector<epicsmap>& bit2SrcVec, vector<unsigned>& epicsmask) {
    // put the epics offsets in the sorted index table
    vector<EpicsInfo> einfo(bit2SrcVec[0].size());
    for (vector<IndexEvent>::iterator itev = _idx.begin(); itev != _idx.end(); ++ itev) {
      unsigned detmask = ~itev->entry.uMaskDetData;
      int file = itev->file;
      if (detmask & epicsmask[file]) {
        // found an event with epics data, update the epics offsets
        for (epicsmap::const_iterator itsrc=bit2SrcVec[file].begin(); itsrc!=bit2SrcVec[file].end(); ++itsrc) {
          if (itsrc->first & detmask) {
            Pds::Src src = itsrc->second;
            einfo[src2EpicsArray[src]].offset = itev->entry.i64OffsetXtc;
            einfo[src2EpicsArray[src]].file = itev->file;
          }
        }
      }
      int i=0;
      // put the latest epics offsets in the official table
      for (vector<EpicsInfo>::iterator it = einfo.begin(); it != einfo.end();  ++it) {
        itev->einfo[i]=*it; // would be neater with vector, but less performant, I believe
        i++;
      }
    }
  }

public:

  IndexRun(queue<DgramPieces>& queue, const vector<XtcFileName> &xtclist) :
    _xtc(xtclist), _beginrunOffset(0), _lastcalib(0,0), _queue(queue) {

    // store the index files in our table, and get some information about epics
    std::map<Pds::Src,int> src2EpicsArray;
    vector<epicsmap> bit2SrcVec;
    vector<unsigned> epicsmask;
    _storeIndex(xtclist,src2EpicsArray,bit2SrcVec,epicsmask);
    
    // update our table with the pointers to the appropriate epics event
    _updateEpics(src2EpicsArray, bit2SrcVec, epicsmask);

    // fill in the list of unique event times that the users will use to jump()
    _fillTimes();
    // send a configure transition
    _configure();
    // send a beginrun transition
    _beginrun();
  }

  ~IndexRun() {}

  // return vector of times that can be used for the "jump" method.
  const vector<psana::EventTime>& times() const {return _times;}

  // jump to an event
  // can't be a const method because it changes the "pieces" object
  int jump(uint64_t timestamp, uint32_t fiducial) {
    uint32_t seconds= (uint32_t)((timestamp&0xffffffff00000000)>>32);
    uint32_t nanoseconds= (uint32_t)(timestamp&0xffffffff);
    IndexEvent request(seconds,nanoseconds,fiducial);
    vector<IndexEvent>::iterator it;
    it = lower_bound(_idx.begin(),_idx.end(),request);
    Pds::Dgram* dg=0;
    if (*it==request) {
      _maybePostCalib(seconds,nanoseconds);
      _pieces.reset();
      // event-build split-events
      int ifirst = 1;
      while (it!=_idx.end() && *it==request) {
        dg = _xtc.jump((*it).file, (*it).entry.i64OffsetXtc);
        _add(dg);
        if (ifirst) {
          ifirst = 0;
          _maybeAddEpics(*it,_epicsSource,dg);
        }
        it++;
      }
      _maybeAddIoc(seconds,fiducial);
      _post();
      return 0;
    } else {
      return 1;
    }
  }

private:
  IndexXtcReader           _xtc;
  vector<IndexEvent>       _idx;
  vector<IndexFiducial>    _idxioc;
  vector<Pds::Src>         _epicsSource;
  vector<IndexCalib>       _idxcalib;
  int64_t                  _beginrunOffset;
  IndexCalib               _lastcalib;
  vector<EpicsInfo>        _lastEpics;
  DgramPieces              _pieces;
  vector<psana::EventTime> _times;
  queue<DgramPieces>&      _queue;
};

// above is the "private" implementation (class IndexRun), below this is the
// "public" implementation (class Index)

Index::Index(const std::string& name, std::queue<DgramPieces>& queue) : Configurable(name), _queue(queue),_idxrun(0) {
  _fileNames = configList("files");
  if ( _fileNames.empty() ) MsgLog(logger, fatal, "Empty file list");
  _rmap = new RunMap(_fileNames);
}

Index::~Index() {
  delete _rmap;
}

int Index::jump(psana::EventTime time) {
  return _idxrun->jump(time.time(),time.fiducial());
}

const vector<psana::EventTime>& Index::runtimes() {
  return _idxrun->times();
}

void Index::setrun(int run) {
  if (not _rmap->runFiles.count(run)) MsgLog(logger, fatal, "Run " << run << " not found");
  delete _idxrun;
  _idxrun = new IndexRun(_queue,_rmap->runFiles[run]);
}

const std::vector<unsigned>& Index::runs() {
  return _rmap->runs;
}

} // namespace PSXtcInput
