//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2ONexusWriter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2ONexusWriter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>
#include <vector>
#include <memory>
#include <cstdio>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/acqiris/ConfigV1.hh"
#include "pdsdata/acqiris/DataDescV1.hh"
#include "pdsdata/evr/ConfigV1.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace nexuspp ;
using namespace O2OTranslator ;

namespace {

  const char* logger = "NexusWriter" ;

  // printable state name
  const char* stateName ( O2ONexusWriter::State state ) {
    switch ( state ) {
      case O2ONexusWriter::Undefined :
        return "Undefined" ;
      case O2ONexusWriter::Mapped :
        return "Mapped" ;
      case O2ONexusWriter::Configured :
        return "Configured" ;
      case O2ONexusWriter::Running :
        return "Running" ;
    }
    return "*ERROR*" ;
  }

  // Class name for new groups depends on the state
  const char* className ( O2ONexusWriter::State state ) {
    switch ( state ) {
      case O2ONexusWriter::Undefined :
        return "NXentry" ;
      case O2ONexusWriter::Mapped :
        return "NXinstrument" ;
      case O2ONexusWriter::Configured :
        return "CfgObject" ;
      case O2ONexusWriter::Running :
        return "NXevent_data" ;
    }
    return "Unexpected" ;
  }

  // cast to DetInfo or throw
  const Pds::DetInfo& detInfo( const Pds::Src& src, const char* cname ) {
    // we have to be at Source level
    if ( src.level() != Pds::Level::Source ) {
      throw O2OXTCLevelException ( cname, Pds::Level::name(src.level()) ) ;
    }

    // can cast to DetInfo now
    return static_cast<const Pds::DetInfo&>(src);
  }

  // get the group name from the source
  std::string groupName( const Pds::DetInfo& info ) {
    std::ostringstream grp ;
    grp << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
        << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;

    return grp.str() ;
  }

  // create new group in a file
  void createGroup ( NxppFile* m_file, const std::string& groupName, O2ONexusWriter::State state ) {
    const char* gclass = className(state) ;
    // create NXentry group
    MsgLog( logger, debug, "O2ONexusWriter -- creating group " << groupName << "/" << gclass ) ;
    m_file->createGroup ( groupName, gclass ) ;
  }

  // store clock value in the attributes of current object
  void storeClock ( NxppFile* m_file, const Pds::ClockTime& clock ) {
    // store transition time as couple of attributes
    m_file->addAttribute ( "time.sec", clock.seconds() ) ;
    m_file->addAttribute ( "time.nsec", clock.nanoseconds() ) ;
  }

  // create and store data entry (single item)
  template <typename T>
  void storeDataSet ( NxppFile* m_file, const char* name, const T& val ) {
    int dims[] = { 1 } ;
    std::auto_ptr<NxppDataSet<T> > ds ( m_file->makeDataSet<T>( name, 1, dims ) ) ;
    ds->putData ( val ) ;
  }

  // create and store data entry (1-dim array)
  template <typename T>
  void storeDataSet ( NxppFile* m_file, const char* name, int size, const T* val ) {
    int dims[] = { size } ;
    std::auto_ptr<NxppDataSet<T> > ds ( m_file->makeDataSet<T>( name, 1, dims ) ) ;
    ds->putData ( val ) ;
  }

  // extend data entry
  template <typename T>
  void extendDataSet ( NxppFile* m_file, const char* name, const T* val ) {
    // open data set first
    std::auto_ptr<NxppDataSet<T> > ds ( m_file->openDataSet<T>( name ) ) ;

    // get its current dimensions
    int rank, dims[NX_MAXRANK] ;
    ds->getInfo( &rank, dims ) ;
    MsgLog(logger,debug, "extendDataSet, rank=" << rank << " dims="<<dims[0]<<','<<dims[1]<<','<<dims[2]<<','<<dims[3] ) ;

    // get the size of the slab
    int size[32] = { 1 } ;
    for ( int i = 1 ; i < rank ; ++ i ) {
      size[i] = dims[i] ;
      dims[i] = 0 ;
    }

    // extend it
    ds->putSlab ( val, dims, size ) ;
  }

  // I want to redefine NeXus error reporter because it's too smart
  void nexusWriterReportError(void *pData, char *string)
  {
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparator for DetInfo objects
bool
O2ONexusWriter::CmpDetInfo::operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const
{
  int ldet = lhs.detector();
  int rdet = rhs.detector();
  if ( ldet < rdet ) return true ;
  if ( ldet > rdet ) return false ;

  int ldev = lhs.device();
  int rdev = rhs.device();
  if ( ldev < rdev ) return true ;
  if ( ldev > rdev ) return false ;

  uint32_t ldetId = lhs.detId();
  uint32_t rdetId = rhs.detId();
  if ( ldetId < rdetId ) return true ;
  if ( ldetId > rdetId ) return false ;

  uint32_t ldevId = lhs.devId();
  uint32_t rdevId = rhs.devId();
  if ( ldevId < rdevId ) return true ;
  if ( ldevId > rdevId ) return false ;

  uint32_t lpid = lhs.processId();
  uint32_t rpid = rhs.processId();
  if ( lpid < rpid ) return true ;
  return false ;
}

//----------------
// Constructors --
//----------------
O2ONexusWriter::O2ONexusWriter ( const O2OFileNameFactory& nameFactory )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file(0)
  , m_state(Undefined)
  , m_acqConfigMap()
{
  // reset NeXus error "handler"
  NXMSetError ( 0, ::nexusWriterReportError ) ;

  // open the NeXus file
  std::string fileName = m_nameFactory.makePath ( 1 ) ;
  MsgLog( logger, debug, "O2ONexusWriter - open output file " << fileName ) ;
  m_file = nexuspp::NxppFile::open ( fileName.c_str(), nexuspp::NxppFile::CreateHdf5 );
  if ( not m_file ) {
    throw O2OFileOpenException(fileName) ;
  }

}

//--------------
// Destructor --
//--------------
O2ONexusWriter::~O2ONexusWriter ()
{
  MsgLog( logger, debug, "O2ONexusWriter - close output file" ) ;

  // this guy could throw
  m_file->close() ;
  delete m_file ;
}

// signal start/end of the event (datagram)
void
O2ONexusWriter::eventStart ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2ONexusWriter::eventStart " << Pds::TransitionId::name(seq.service())
          << " seq.type=" << seq.type()
          << " seq.service=" << Pds::TransitionId::name(seq.service()) ) ;

  if ( seq.service() == Pds::TransitionId::Map ) {

    // check current state
    if ( m_state != Undefined ) {
      throw O2OXTCTransitionException( "Map", ::stateName(m_state) ) ;
    }

    // dump seconds as a hex string, it will be group name
    char buf[32] ;
    snprintf ( buf, sizeof buf, "Map:%08X", seq.clock().seconds() ) ;

    // create NXentry group
    createGroup ( m_file, buf, m_state ) ;

    // store transition time as couple of attributes to this new group
    ::storeClock ( m_file, seq.clock() ) ;

    // switch to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Configure ) {

    // check current state
    if ( m_state != Mapped ) {
      throw O2OXTCTransitionException( "Configure", ::stateName(m_state) ) ;
    }

    // Create 'NXinstrument' group
    createGroup ( m_file, "Configure", m_state ) ;

    // store transition time as couple of attributes to this new group
    ::storeClock ( m_file, seq.clock() ) ;

    // switch to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::BeginRun ) {

    // check current state
    if ( m_state != Configured ) {
      throw O2OXTCTransitionException( "BeginRun", ::stateName(m_state) ) ;
    }

    // switch to running state
    m_state = Running ;

  } else if ( seq.service() == Pds::TransitionId::EndRun ) {

    // switch back to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::Unconfigure ) {

    // switch back to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Unmap ) {

    // close 'NXentry' group, go back to top level
    m_file->closeGroup() ;

    // switch back to undetermined state
    m_state = Undefined ;

  }

  MsgLog( logger, debug, "O2ONexusWriter -- now in the state " << ::stateName(m_state) ) ;
}

void
O2ONexusWriter::eventEnd ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2ONexusWriter::eventEnd " << Pds::TransitionId::name(seq.service()) ) ;

  if ( seq.service() == Pds::TransitionId::Configure ) {

    // check current state
    if ( m_state != Configured ) {
      throw O2OXTCTransitionException( "Configure", ::stateName(m_state) ) ;
    }

    // close 'NXinstrument' group, go back one level
    m_file->closeGroup() ;
  }

}

// signal start/end of the level
void
O2ONexusWriter::levelStart ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::levelStart " << Pds::Level::name(src.level()) ) ;
}

void
O2ONexusWriter::levelEnd ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::levelEnd " << Pds::Level::name(src.level()) ) ;
}

// visit the data object
void
O2ONexusWriter::dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src )
{
  static const char cname[] = "Acqiris::ConfigV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create object group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;

  // split the data
  uint32_t nChan = data.nbrChannels() ;
  double fullScale[Pds::Acqiris::ConfigV1::MaxChan] ;
  double offset[Pds::Acqiris::ConfigV1::MaxChan] ;
  uint32_t coupling[Pds::Acqiris::ConfigV1::MaxChan] ;
  uint32_t bandwidth[Pds::Acqiris::ConfigV1::MaxChan] ;
  for ( uint32_t i = 0 ; i < nChan ; ++ i ) {
    const Pds::Acqiris::VertV1& vconfig = data.vert(i) ;
    fullScale[i] = vconfig.fullScale() ;
    offset[i] = vconfig.offset() ;
    coupling[i] = vconfig.coupling() ;
    bandwidth[i] = vconfig.bandwidth() ;
  }
  const Pds::Acqiris::HorizV1& hconfig = data.horiz() ;
  const Pds::Acqiris::TrigV1& trigConfig = data.trig() ;


  ::storeDataSet ( m_file, "nbrConvertersPerChannel", data.nbrConvertersPerChannel() ) ;
  ::storeDataSet ( m_file, "channelMask",   data.channelMask() ) ;
  ::storeDataSet ( m_file, "nbrChannels",   data.nbrChannels() ) ;
  ::storeDataSet ( m_file, "nbrBanks",      data.nbrBanks() ) ;

  ::storeDataSet ( m_file, "vert.fullScale", nChan, fullScale ) ;
  ::storeDataSet ( m_file, "vert.offset",    nChan, offset ) ;
  ::storeDataSet ( m_file, "vert.coupling",  nChan, coupling ) ;
  ::storeDataSet ( m_file, "vert.bandwidth", nChan, bandwidth ) ;

  ::storeDataSet ( m_file, "horiz.sampInterval",  hconfig.sampInterval() ) ;
  ::storeDataSet ( m_file, "horiz.delayTime",     hconfig.delayTime() ) ;
  ::storeDataSet ( m_file, "horiz.nbrSamples",    hconfig.nbrSamples() ) ;
  ::storeDataSet ( m_file, "horiz.nbrSegments",   hconfig.nbrSegments() ) ;

  ::storeDataSet ( m_file, "trig.trigCoupling",  trigConfig.trigCoupling() ) ;
  ::storeDataSet ( m_file, "trig.trigInput",     trigConfig.trigInput() ) ;
  ::storeDataSet ( m_file, "trig.trigSlope",     trigConfig.trigSlope() ) ;
  ::storeDataSet ( m_file, "trig.trigLevel",     trigConfig.trigLevel() ) ;

  // go up one level
  m_file->closeGroup () ;

  // store also the copy for later
  m_acqConfigMap.insert( AcqConfigMap::value_type(detInfo,data) ) ;
}

void
O2ONexusWriter::dataObject ( const Pds::Acqiris::DataDescV1& data, const Pds::Src& src )
{
  static const char cname[] = "Acqiris::DataDescV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // to interpret the content we need corresponding Config object
  AcqConfigMap::const_iterator cfgIt = m_acqConfigMap.find ( detInfo ) ;
  if ( cfgIt == m_acqConfigMap.end() ) {
    throw O2OXTCConfigException ( "Acqiris::ConfigV1" ) ;
  }
  const Pds::Acqiris::ConfigV1& config = cfgIt->second ;
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;

  // Create or open object group
  bool created = false ;
  try {
    m_file->openGroup ( grp, ::className(m_state) ) ;
  } catch ( NxppException& e ) {
    // failed to open, try to create
    createGroup ( m_file, grp, m_state ) ;
    m_file->addAttribute ( "version", 1 ) ;
    created = true ;
  }

  // get few constants
  const uint32_t nChan = config.nbrChannels() ;
  const uint32_t nSeg = hconfig.nbrSegments() ;
  const uint32_t nSampl = hconfig.nbrSamples() ;

  // allocate data
  uint64_t timestamps[nChan][nSeg] ;
  uint16_t waveforms[nChan][nSeg][nSampl] ;

  // scan the data and fill arrays
  // FIXME: few methods that we need from DataDescV1 declared as non-const
  Pds::Acqiris::DataDescV1& dd = const_cast<Pds::Acqiris::DataDescV1&>( data ) ;
  for ( uint32_t ch = 0 ; ch < nChan ; ++ ch, dd = *dd.nextChannel(hconfig) ) {

    // first verify that the shape of the data returned corresponds to the config
    if ( dd.nbrSamplesInSeg() != nSampl ) {
      std::ostringstream msg ;
      msg << "O2ONexusWriter::dataObject(Acqiris::DataDescV1) -"
          << " number of samples in data object (" << dd.nbrSamplesInSeg()
          << ") different from config object (" << nSampl << ")" ;
      throw O2OXTCGenException ( msg.str() ) ;
    }
    if ( dd.nbrSegments() != nSeg ) {
      std::ostringstream msg ;
      msg << "O2ONexusWriter::dataObject(Acqiris::DataDescV1) -"
          << " number of segments in data object (" << dd.nbrSegments()
          << ") different from config object (" << nSeg << ")" ;
      throw O2OXTCGenException ( msg.str() ) ;
    }

    for ( uint32_t seg = 0 ; seg < nSeg ; ++ seg ) {
      timestamps[ch][seg] = dd.timestamp(seg).value();
    }

    uint16_t* wf = dd.waveform(hconfig) ;
    std::copy ( wf, wf+nSampl*nSeg, (uint16_t*)waveforms[ch] ) ;
  }

  if ( created ) {
    // create data items if needed
    int ts_dims[] = { NX_UNLIMITED, nChan, nSeg } ;
    NxppDataSet<uint64_t>* ds_ts = m_file->makeDataSet<uint64_t>( "timestamp", 3, ts_dims ) ;
    delete ds_ts ;
    int wf_dims[] = { NX_UNLIMITED, nChan, nSeg, nSampl } ;
    NxppDataSet<uint16_t>* ds_wf = m_file->makeDataSet<uint16_t>( "waveform", 4, wf_dims ) ;
    delete ds_wf ;
  }

  // store the data
  ::extendDataSet ( m_file, "timestamp", (uint64_t*)timestamps ) ;
  ::extendDataSet ( m_file, "waveform", (uint16_t*)waveforms ) ;

  // go up one level
  m_file->closeGroup () ;

}

void
O2ONexusWriter::dataObject ( const Pds::Camera::FrameFexConfigV1& data, const Pds::Src& src )
{
  static const char cname[] = "Camera::FrameFexConfigV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create object group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;


  // go up one level
  m_file->closeGroup () ;

}

void
O2ONexusWriter::dataObject ( const Pds::Camera::FrameV1& data, const Pds::Src& src )
{
  static const char cname[] = "Camera::FrameV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create object group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;


  // go up one level
  m_file->closeGroup () ;

}

void
O2ONexusWriter::dataObject ( const Pds::Camera::TwoDGaussianV1& data, const Pds::Src& src )
{
  static const char cname[] = "Camera::TwoDGaussianV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create object group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;


  // go up one level
  m_file->closeGroup () ;

}

void
O2ONexusWriter::dataObject ( const Pds::EvrData::ConfigV1& data, const Pds::Src& src )
{
  static const char cname[] = "EvrData::ConfigV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create object group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;


  // store all pulses configurations
  unsigned npulses = data.npulses() ;
  ::storeDataSet ( m_file, "npulses", npulses );

  if ( npulses ) {

    std::vector<unsigned> v_pulse(npulses,0);
    std::vector<int> v_trigger(npulses,0);
    std::vector<int> v_set(npulses,0);
    std::vector<int> v_clear(npulses,0);
    std::vector<uint8_t> v_polarity(npulses,0);
    std::vector<uint8_t> v_map_set_enable(npulses,0);
    std::vector<uint8_t> v_map_reset_enable(npulses,0);
    std::vector<uint8_t> v_map_trigger_enable(npulses,0);
    std::vector<unsigned> v_prescale(npulses,0);
    std::vector<unsigned> v_delay(npulses,0);
    std::vector<unsigned> v_width(npulses,0);

    for ( unsigned i = 0 ; i < npulses ; ++ i ) {
      const Pds::EvrData::PulseConfig& pconf = data.pulse(i) ;
      v_pulse[i] = pconf.pulse() ;
      v_trigger[i] = pconf.trigger() ;
      v_set[i] = pconf.set() ;
      v_clear[i] = pconf.clear() ;
      v_polarity[i] = pconf.polarity() ;
      v_map_set_enable[i] = pconf.map_set_enable() ;
      v_map_reset_enable[i] = pconf.map_reset_enable() ;
      v_map_trigger_enable[i] = pconf.map_trigger_enable() ;
      v_prescale[i] = pconf.prescale() ;
      v_delay[i] = pconf.delay() ;
      v_width[i] = pconf.delay() ;
    }

    ::storeDataSet ( m_file, "pulse.pulse", npulses, &v_pulse.front() );
    ::storeDataSet ( m_file, "pulse.trigger", npulses, &v_trigger.front() );
    ::storeDataSet ( m_file, "pulse.set", npulses, &v_set.front() );
    ::storeDataSet ( m_file, "pulse.clear", npulses, &v_clear.front() );
    ::storeDataSet ( m_file, "pulse.polarity", npulses, &v_polarity.front() );
    ::storeDataSet ( m_file, "pulse.map_set_enable", npulses, &v_map_set_enable.front() );
    ::storeDataSet ( m_file, "pulse.map_reset_enable", npulses, &v_map_reset_enable.front() );
    ::storeDataSet ( m_file, "pulse.map_trigger_enable", npulses, &v_map_trigger_enable.front() );
    ::storeDataSet ( m_file, "pulse.prescale", npulses, &v_prescale.front() );
    ::storeDataSet ( m_file, "pulse.delay", npulses, &v_delay.front() );
    ::storeDataSet ( m_file, "pulse.width", npulses, &v_width.front() );

  }


  // store all outputs configurations
  unsigned noutputs = data.noutputs() ;
  ::storeDataSet ( m_file, "noutputs", noutputs );

  if ( noutputs ) {
    std::vector<unsigned> v_source( noutputs,0 ) ;
    std::vector<unsigned> v_source_id( noutputs,0 ) ;
    std::vector<unsigned> v_conn( noutputs,0 ) ;
    std::vector<unsigned> v_conn_id( noutputs,0 ) ;

    for ( unsigned i = 0 ; i < noutputs ; ++ i ) {
      const Pds::EvrData::OutputMap& omap = data.output_map(i) ;
      v_source[i] = omap.source() ;
      v_source_id[i] = omap.source_id() ;
      v_conn[i] = omap.conn() ;
      v_conn_id[i] = omap.conn_id() ;
    }

    ::storeDataSet ( m_file, "output_map.source", noutputs, &v_source.front() );
    ::storeDataSet ( m_file, "output_map.source_id", noutputs, &v_source_id.front() );
    ::storeDataSet ( m_file, "output_map.conn", noutputs, &v_conn.front() );
    ::storeDataSet ( m_file, "output_map.conn_id", noutputs, &v_conn_id.front() );
  }


  // go up one level
  m_file->closeGroup () ;

}

void
O2ONexusWriter::dataObject ( const Pds::Opal1k::ConfigV1& data, const Pds::Src& src )
{
  static const char cname[] = "Opal1k::ConfigV1" ;

  MsgLog( logger, debug, "O2ONexusWriter::dataObject " << cname << " " << Pds::Level::name(src.level()) ) ;

  // get the DetInfo and group name, this can throw
  const Pds::DetInfo& detInfo = ::detInfo( src, cname );
  const std::string& grp = ::groupName( detInfo ) ;

  // Create 'Object' group
  createGroup ( m_file, grp, m_state ) ;
  m_file->addAttribute ( "version", 1 ) ;


  // go up one level
  m_file->closeGroup () ;

}


} // namespace O2OTranslator
