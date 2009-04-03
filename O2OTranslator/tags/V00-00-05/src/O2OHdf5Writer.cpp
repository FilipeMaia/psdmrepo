//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OHdf5Writer...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OHdf5Writer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

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

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace O2OTranslator ;

namespace {

  const char* logger = "HDF5Writer" ;

  // printable state name
  const char* stateName ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "Undefined" ;
      case O2OHdf5Writer::Mapped :
        return "Mapped" ;
      case O2OHdf5Writer::Configured :
        return "Configured" ;
      case O2OHdf5Writer::Running :
        return "Running" ;
    }
    return "*ERROR*" ;
  }

  // Class name for new groups depends on the state
  const char* className ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "NXentry" ;
      case O2OHdf5Writer::Mapped :
        return "NXinstrument" ;
      case O2OHdf5Writer::Configured :
        return "CfgObject" ;
      case O2OHdf5Writer::Running :
        return "NXevent_data" ;
    }
    return "Unexpected" ;
  }

  // create new group in a file
  hdf5pp::Group createGroup ( hdf5pp::File file, const std::string& groupName, O2OHdf5Writer::State state ) {
    const char* gclass = className(state) ;
    MsgLog( logger, debug, "HDF5Writer -- creating group " << groupName << "/" << gclass ) ;
    hdf5pp::Group grp = file.createGroup( groupName ) ;

    // make an attribute
    hdf5pp::Attribute<const char*> clsAttr = grp.createAttr<const char*> ( "NX_class" ) ;
    clsAttr.store ( gclass ) ;

    return grp ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparator for DetInfo objects
bool
O2OHdf5Writer::CmpDetInfo::operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const
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
O2OHdf5Writer::O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                               bool overwrite,
                               SplitMode split,
                               hsize_t splitSize )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file()
  , m_state(Undefined)
  , m_acqConfigMap()
{
  std::string fileTempl = m_nameFactory.makeH5Path () ;
  MsgLog( logger, debug, "O2OHdf5Writer - open output file " << fileTempl ) ;

  // Disable printing of error messages
  //stat = H5Eset_auto2( H5E_DEFAULT, 0, 0 ) ;

  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( split == Family ) {
    // use FAMILY driver
    fapl.set_family_driver ( splitSize, hdf5pp::PListFileAccess() ) ;
  }

  hdf5pp::File::CreateMode mode = overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;
  m_file = hdf5pp::File::create ( fileTempl, mode, hdf5pp::PListFileCreate(), fapl ) ;

}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog( logger, debug, "O2OHdf5Writer - close output file" ) ;
  // destructor will close the file
}

// signal start/end of the event (datagram)
void
O2OHdf5Writer::eventStart ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(seq.service())
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
    //::storeClock ( m_file, seq.clock() ) ;

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
    //::storeClock ( m_file, seq.clock() ) ;

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
    //m_file->closeGroup() ;

    // switch back to undetermined state
    m_state = Undefined ;

  }

  MsgLog( logger, debug, "O2OHdf5Writer -- now in the state " << ::stateName(m_state) ) ;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventEnd " << Pds::TransitionId::name(seq.service()) ) ;

}

// signal start/end of the level
void
O2OHdf5Writer::levelStart ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelStart " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::levelEnd ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelEnd " << Pds::Level::name(src.level()) ) ;
}

// visit the data object
void
O2OHdf5Writer::dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Acqiris::ConfigV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Acqiris::DataDescV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Acqiris::DataDescV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::FrameFexConfigV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Camera::FrameFexConfigV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::FrameV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Camera::FrameV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::TwoDGaussianV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Camera::TwoDGaussianV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::EvrData::ConfigV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject EvrData::ConfigV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Opal1k::ConfigV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::dataObject Opal1k::ConfigV1 " << Pds::Level::name(src.level()) ) ;
}

} // namespace O2OTranslator
