//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/PnCCDFrameV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "O2OTranslator/O2OExceptions.h"
#include "MsgLogger/MsgLogger.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "PnCCDFrameV1Cvt" ;
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparison operator for Src objects
bool
PnCCDFrameV1Cvt::_SrcCmp::operator()( const Pds::Src& lhs, const Pds::Src& rhs ) const
{
  if ( lhs.log() < rhs.log() ) return true ;
  if ( lhs.log() > rhs.log() ) return false ;
  if ( lhs.phy() < rhs.phy() ) return true ;
  return false ;
}

//----------------
// Constructors --
//----------------
PnCCDFrameV1Cvt::PnCCDFrameV1Cvt ( const std::string& typeGroupName,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::PNCCD::FrameV1>(typeGroupName)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_config()
  , m_frameCont(0)
  , m_frameDataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
PnCCDFrameV1Cvt::~PnCCDFrameV1Cvt ()
{
  delete m_frameCont ;
  delete m_frameDataCont ;
  delete m_timeCont ;
}

/// override base class method because we expect multiple types
void
PnCCDFrameV1Cvt::convert ( const void* data,
                           size_t size,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src,
                           const H5DataTypes::XtcClockTime& time )
{
  if ( typeId.id() == Pds::TypeId::Id_pnCCDconfig ) {

    const ConfigXtcType& config = *static_cast<const ConfigXtcType*>( data ) ;

    // check data size
    if ( sizeof(ConfigXtcType) != size ) {
      throw O2OXTCSizeException ( "PNCCD::ConfigV1", sizeof(ConfigXtcType), size ) ;
    }

    // got configuration object, make the copy
    m_config.insert( ConfigMap::value_type( src.top(), config ) ) ;

  } else if ( typeId.id() == Pds::TypeId::Id_pnCCDframe ) {

    // follow regular path
    const XtcType& typedData = *static_cast<const XtcType*>( data ) ;

    typedConvert ( typedData, size, typeId, src, time ) ;

  }

}

// typed conversion method
void
PnCCDFrameV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                        const XtcType& data,
                                        size_t size,
                                        const Pds::TypeId& typeId,
                                        const O2OXtcSrc& src,
                                        const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  ConfigMap::const_iterator cit = m_config.find( src.top() ) ;
  if ( cit == m_config.end() ) {
    MsgLog ( logger, error, "PnCCDFrameV1Cvt - no configuration object was defined" );
    return ;
  }
  const ConfigXtcType& config = cit->second ;

  // create all containers if running first time
  if ( not m_frameCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::PnCCDFrameV1> frContFactory( "frame", m_chunk_size, m_deflate ) ;
    m_frameCont = new FrameCont ( frContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate ) ;
    m_frameDataCont = new FrameDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }


  // get few constants
  const uint32_t numLinks = config.numLinks() ;
  const unsigned sizeofData = data.sizeofData(config) ;

  // make data arrays
  H5DataTypes::PnCCDFrameV1 frame[numLinks] ;
  uint16_t frameData[numLinks][sizeofData] ;

  // move the data
  const XtcType* dd = &data ;
  for ( unsigned link = 0 ; link != numLinks ; ++ link ) {

    // copy frame info
    frame[link] = H5DataTypes::PnCCDFrameV1(*dd) ;

    // copy data
    const uint16_t* ddata = dd->data() ;
    std::copy( ddata, ddata+sizeofData, (uint16_t*)frameData[link] ) ;

    // move to next frame
    dd = dd->next(config) ;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::PnCCDFrameV1::stored_type ( config ) ;
  m_frameCont->container(group,type)->append ( frame[0], type ) ;
  type = H5DataTypes::PnCCDFrameV1::stored_data_type ( config ) ;
  m_frameDataCont->container(group,type)->append ( frameData[0][0], type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
PnCCDFrameV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_frameCont ) m_frameCont->closeGroup( group ) ;
  if ( m_frameDataCont ) m_frameDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
