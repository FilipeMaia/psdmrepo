//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/AcqirisDataDescV1.h"
#include "O2OTranslator/O2OExceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "AcqirisDataDescV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparison operator for Src objects
bool
AcqirisDataDescV1Cvt::_SrcCmp::operator()( const Pds::Src& lhs, const Pds::Src& rhs ) const
{
  if ( lhs.log() < rhs.log() ) return true ;
  if ( lhs.log() > rhs.log() ) return false ;
  if ( lhs.phy() < rhs.phy() ) return true ;
  return false ;
}

//----------------
// Constructors --
//----------------
AcqirisDataDescV1Cvt::AcqirisDataDescV1Cvt ( const std::string& typeGroupName,
                                             hsize_t chunk_size,
                                             int deflate )
  : EvtDataTypeCvt<Pds::Acqiris::DataDescV1>(typeGroupName)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_config()
  , m_timestampCont(0)
  , m_waveformCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisDataDescV1Cvt::~AcqirisDataDescV1Cvt ()
{
  delete m_timestampCont ;
  delete m_waveformCont ;
  delete m_timeCont ;
}

/// override base class method because we expect multiple types
void
AcqirisDataDescV1Cvt::convert ( const void* data,
                                size_t size,
                                const Pds::TypeId& typeId,
                                const O2OXtcSrc& src,
                                const H5DataTypes::XtcClockTime& time )
{
  if ( typeId.id() == Pds::TypeId::Id_AcqConfig ) {

    const ConfigXtcType& config = *static_cast<const ConfigXtcType*>( data ) ;

    // check data size
    if ( sizeof(ConfigXtcType) != size ) {
      throw O2OXTCSizeException ( "Acqiris::ConfigV1", sizeof(ConfigXtcType), size ) ;
    }
    
    // got configuration object, make the copy
    m_config.insert( ConfigMap::value_type( src.top(), config ) ) ;

  } else if ( typeId.id() == Pds::TypeId::Id_AcqWaveform ) {

    // follow regular path
    const XtcType& typedData = *static_cast<const XtcType*>( data ) ;

    typedConvert ( typedData, size, typeId, src, time ) ;

  }

}

// typed conversion method
void
AcqirisDataDescV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                             const XtcType& data,
                                             size_t size,
                                             const Pds::TypeId& typeId,
                                             const O2OXtcSrc& src,
                                             const H5DataTypes::XtcClockTime& time )
{
      
  // find corresponding configuration object
  ConfigMap::const_iterator cit = m_config.find( src.top() ) ;
  if ( cit == m_config.end() ) {
    MsgLog ( logger, error, "AcqirisDataDescV1Cvt - no configuration object was defined" );
  }
  const ConfigXtcType& config = cit->second ;

  // create all containers if running first time
  if ( not m_waveformCont ) {

    // create container for timestamps
    CvtDataContFactoryAcqirisV1<uint64_t> tsContFactory( "timestamps", m_chunk_size, m_deflate, 'T' ) ;
    m_timestampCont = new TimestampCont ( tsContFactory ) ;

    // create container for waveforms
    CvtDataContFactoryAcqirisV1<uint16_t> wfContFactory( "waveforms", m_chunk_size, m_deflate, 'W' ) ;
    m_waveformCont = new WaveformCont ( wfContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }


  // get few constants
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;
  const uint32_t nChan = config.nbrChannels() ;
  const uint32_t nSeg = hconfig.nbrSegments() ;
  const uint32_t nSampl = hconfig.nbrSamples() ;

  // check total size
  if ( data.totalSize(hconfig) * nChan != size ) {
    throw O2OXTCSizeException ( "AcqirisDataDescV1", data.totalSize(hconfig) * nChan, size ) ;
  }

  // allocate data
  uint64_t timestamps[nChan][nSeg] ;
  uint16_t waveforms[nChan][nSeg][nSampl] ;

  // scan the data and fill arrays
  // FIXME: few methods that we need from DataDescV1 declared as non-const
  Pds::Acqiris::DataDescV1* dd = const_cast<Pds::Acqiris::DataDescV1*>( &data ) ;
  for ( uint32_t ch = 0 ; ch < nChan ; ++ ch, dd = dd->nextChannel(hconfig) ) {

    // first verify that the shape of the data returned corresponds to the config
    if ( dd->nbrSamplesInSeg() != nSampl ) {
      std::ostringstream msg ;
      msg << "Acqiris::DataDescV1 - number of samples in data object (" 
          << dd->nbrSamplesInSeg() << ") different from config object (" << nSampl << ")" ;
      throw O2OXTCGenException ( msg.str() ) ;
    }
    if ( dd->nbrSegments() != nSeg ) {
      std::ostringstream msg ;
      msg << "Acqiris::DataDescV1 - number of segments in data object (" 
          << dd->nbrSegments() << ") different from config object (" << nSeg << ")" ;
      throw O2OXTCGenException ( msg.str() ) ;
    }

    for ( uint32_t seg = 0 ; seg < nSeg ; ++ seg ) {
      timestamps[ch][seg] = dd->timestamp(seg).value();
    }

    int16_t* wf = dd->waveform(hconfig) ;
    std::copy ( wf, wf+nSampl*nSeg, (int16_t*)waveforms[ch] ) ;
  }


  // store the data
  hdf5pp::Type type = H5DataTypes::AcqirisDataDescV1::timestampType ( config ) ;
  m_timestampCont->container(group,config)->append ( timestamps[0][0], type ) ;
  type = H5DataTypes::AcqirisDataDescV1::waveformType ( config ) ;
  m_waveformCont->container(group,config)->append ( waveforms[0][0][0], type ) ;
  m_timeCont->container(group)->append ( time ) ;


}

/// method called when the driver closes a group in the file
void
AcqirisDataDescV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_timestampCont ) m_timestampCont->closeGroup( group ) ;
  if ( m_waveformCont ) m_waveformCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}



} // namespace O2OTranslator
