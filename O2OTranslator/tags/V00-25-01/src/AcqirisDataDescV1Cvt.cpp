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

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <sstream>
#include <limits>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/acqiris/ConfigV1.hh"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "AcqirisDataDescV1Cvt" ;

  unsigned warning_count = 10;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
AcqirisDataDescV1Cvt::AcqirisDataDescV1Cvt ( const hdf5pp::Group& group,
    const std::string& typeGroupName,
    const Pds::Src& src,
    const ConfigObjectStore& configStore,
    const CvtOptions& cvtOptions )
  : EvtDataTypeCvt<XtcType>(group, typeGroupName, src, cvtOptions)
  , m_configStore(configStore)
  , m_timestampCont()
  , m_waveformCont()
  , n_miss(0)
{
}

//--------------
// Destructor --
//--------------
AcqirisDataDescV1Cvt::~AcqirisDataDescV1Cvt ()
{
}

/// method called to create all necessary data containers
void
AcqirisDataDescV1Cvt::makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // nothing to do here, types depend on actual data
}

// typed conversion method
void
AcqirisDataDescV1Cvt::fillContainers(hdf5pp::Group group,
    const XtcType& data,
    size_t size,
    const Pds::TypeId& typeId,
    const O2OXtcSrc& src)
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_AcqConfig,1);
  const Pds::Acqiris::ConfigV1* config = m_configStore.find<Pds::Acqiris::ConfigV1>(cfgTypeId, src.top());
  if ( not config ) {
    MsgLog ( logger, error, "AcqirisDataDescV1Cvt - no configuration object was defined" );
    return ;
  }

  // get few constants
  const Pds::Acqiris::HorizV1& hconfig = config->horiz() ;
  const uint32_t nChan = config->nbrChannels() ;
  const uint32_t nSeg = hconfig.nbrSegments() ;
  const uint32_t nSampl = hconfig.nbrSamples() ;

  // check total size
  if ( data.totalSize(hconfig) * nChan != size ) {
    throw O2OXTCSizeException ( ERR_LOC, "AcqirisDataDescV1", data.totalSize(hconfig) * nChan, size ) ;
  }

  // allocate data
  uint64_t timestamps[nChan][nSeg] ;
  int16_t waveforms[nChan][nSeg][nSampl] ;

  // scan the data and fill arrays
  // FIXME: few methods that we need from DataDescV1 declared as non-const
  XtcType* dd = const_cast<XtcType*>( &data ) ;
  for ( uint32_t ch = 0 ; ch < nChan ; ++ ch, dd = dd->nextChannel(hconfig) ) {

    // first verify that the shape of the data returned corresponds to the config
    if ( dd->nbrSamplesInSeg() != nSampl ) {
      if (dd->nbrSamplesInSeg() == 0) {
        if ( warning_count > 0) {
          // means there was no data in this channel, will fill it with some nonsensical stuff
          MsgLog(logger, warning, "AcqirisDataDescV1Cvt - no data samples in data object, will fill with constant data");
          -- warning_count;
        }
      } else {
        // if non-zero and not as expected then it is an error
        MsgLog(logger, error, "AcqirisDataDescV1Cvt - number of samples in data object ("
               << dd->nbrSamplesInSeg() << ") different from config object (" << nSampl << ")");
        // stop here
        return;
      }
    }
    if ( dd->nbrSegments() != nSeg ) {
      if (dd->nbrSegments() == 0) {
        if ( warning_count > 0) {
          // means there was no data in this channel, will fill it with some nonsensical stuff
          MsgLog(logger, warning, "AcqirisDataDescV1Cvt - no segments in data object, will fill with constant data");
          -- warning_count;
        }
      } else {
        MsgLog(logger, error, "AcqirisDataDescV1Cvt - number of segments in data object ("
            << dd->nbrSegments() << ") different from config object (" << nSeg << ")");
        // stop here
        return;
      }
    }

    if (dd->nbrSegments() == 0) {
      // fill with zeros
      std::fill_n(timestamps[ch], nSeg, uint64_t(0));
    } else {
      for ( uint32_t seg = 0 ; seg < nSeg ; ++ seg ) {
        timestamps[ch][seg] = dd->timestamp(seg).value();
      }
    }

    if (dd->nbrSegments() == 0 or dd->nbrSamplesInSeg() == 0) {
      std::fill_n(waveforms[ch][0], nSampl*nSeg, int16_t(0));
    } else {
      int16_t* wf = dd->waveform(hconfig) ;
      std::copy ( wf, wf+nSampl*nSeg, waveforms[ch][0] ) ;
    }
  }

  // store the data
  hdf5pp::Type type = H5Type::timestampType ( *config ) ;
  if (not m_timestampCont) {
    m_timestampCont = makeCont<TimestampCont>("timestamps", group, true, type) ;
    if (n_miss) m_timestampCont->resize(n_miss);
  }
  m_timestampCont->append ( timestamps[0][0], type ) ;
  type = H5Type::waveformType ( *config ) ;
  if (not m_waveformCont) {
    m_waveformCont = makeCont<WaveformCont>("waveforms", group, true, type) ;
    if (n_miss) m_waveformCont->resize(n_miss);
  }
  m_waveformCont->append ( waveforms[0][0][0], type ) ;

}

// fill containers for missing data
void
AcqirisDataDescV1Cvt::fillMissing(hdf5pp::Group group,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src)
{
  if (m_timestampCont) {
    m_timestampCont->resize(m_timestampCont->size() + 1);
    m_waveformCont->resize(m_waveformCont->size() + 1);
  } else {
    ++ n_miss;
  }
}

} // namespace O2OTranslator
