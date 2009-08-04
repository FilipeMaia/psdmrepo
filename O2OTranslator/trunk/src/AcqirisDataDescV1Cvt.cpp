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

//----------------
// Constructors --
//----------------
AcqirisDataDescV1Cvt::AcqirisDataDescV1Cvt ( hdf5pp::Group group,
                                             hsize_t chunk_size,
                                             int deflate )
  : DataTypeCvtI()
  , m_group(group)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_config(0)
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
  delete m_config ;
  delete m_timestampCont ;
  delete m_waveformCont ;
  delete m_timeCont ;
}

/// main method of this class
void
AcqirisDataDescV1Cvt::convert ( const void* data,
                                const Pds::TypeId& typeId,
                                const Pds::DetInfo& detInfo,
                                const H5DataTypes::XtcClockTime& time )
{
  if ( typeId.id() == Pds::TypeId::Id_AcqConfig ) {

    const Pds::Acqiris::ConfigV1& config = *static_cast<const Pds::Acqiris::ConfigV1*>( data ) ;

    // got configuration object, make the copy
    delete m_config ;
    m_config = new Pds::Acqiris::ConfigV1( config );

    // create container for timestamps
    m_tsType = H5DataTypes::AcqirisDataDescV1::timestampType ( *m_config ) ;
    hsize_t chunk = std::max( m_chunk_size/m_tsType.size(), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for timestamps: " << chunk ) ;
    m_timestampCont = new TimestampCont ( "timestamps", m_group, m_tsType, chunk, m_deflate ) ;

    // create container for waveforms
    m_wfType = H5DataTypes::AcqirisDataDescV1::waveformType ( *m_config ) ;
    chunk = std::max( m_chunk_size/m_wfType.size(), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for waveforms: " << chunk ) ;
    m_waveformCont = new WaveformCont ( "waveforms", m_group, m_wfType, chunk, m_deflate ) ;

    // make container for time
    chunk = std::max ( m_chunk_size / sizeof(H5DataTypes::XtcClockTime), hsize_t(1) ) ;
    MsgLog( logger, debug, "chunk size for time: " << chunk ) ;
    m_timeCont = new XtcClockTimeCont ( "time", m_group, chunk, m_deflate ) ;

  } else if ( typeId.id() == Pds::TypeId::Id_AcqWaveform ) {

    // get few constants
    const Pds::Acqiris::HorizV1& hconfig = m_config->horiz() ;
    const uint32_t nChan = m_config->nbrChannels() ;
    const uint32_t nSeg = hconfig.nbrSegments() ;
    const uint32_t nSampl = hconfig.nbrSamples() ;

    // allocate data
    uint64_t timestamps[nChan][nSeg] ;
    uint16_t waveforms[nChan][nSeg][nSampl] ;

    // scan the data and fill arrays
    // FIXME: few methods that we need from DataDescV1 declared as non-const
    const Pds::Acqiris::DataDescV1& ddescr = *static_cast<const Pds::Acqiris::DataDescV1*>( data ) ;
    Pds::Acqiris::DataDescV1& dd = const_cast<Pds::Acqiris::DataDescV1&>( ddescr ) ;
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

      int16_t* wf = dd.waveform(hconfig) ;
      std::copy ( wf, wf+nSampl*nSeg, (int16_t*)waveforms[ch] ) ;
    }


    // store the data
    m_timestampCont->append ( timestamps[0][0], m_tsType ) ;
    m_waveformCont->append ( waveforms[0][0][0], m_wfType ) ;
    m_timeCont->append ( time ) ;

  }


}

} // namespace O2OTranslator
