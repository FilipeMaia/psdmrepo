//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV2Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadElementV2Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/ConfigObjectStore.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/cspad/ConfigV1.hh"
#include "pdsdata/cspad/ConfigV2.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPadElementV2Cvt" ;
  
  // slow bit count
  unsigned bitCount(uint32_t mask, unsigned maxBits) {
    unsigned res = 0;
    for (  ; maxBits ; -- maxBits ) {
      if ( mask & 1 ) ++ res ;
      mask >>= 1 ;
    }
    return res ;
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
CsPadElementV2Cvt::CsPadElementV2Cvt ( const std::string& typeGroupName,
                                   const ConfigObjectStore& configStore,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::CsPad::ElementV2>(typeGroupName)
  , m_configStore(configStore)
  , m_chunk_size(chunk_size)
  , m_deflate(deflate)
  , m_elementCont(0)
  , m_pixelDataCont(0)
  , m_timeCont(0)
{
}

//--------------
// Destructor --
//--------------
CsPadElementV2Cvt::~CsPadElementV2Cvt ()
{
  delete m_elementCont ;
  delete m_pixelDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CsPadElementV2Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                          const XtcType& data,
                                          size_t size,
                                          const Pds::TypeId& typeId,
                                          const O2OXtcSrc& src,
                                          const H5DataTypes::XtcClockTime& time )
{
  // based on cspad/ElementIterator but we cannot use that class directly
  uint32_t qMask = 0;
  uint32_t sMask[Pds::CsPad::MaxQuadsPerSensor];
  unsigned sections = 0;
  
  // find corresponding configuration object, it could be ConfigV1 or ConfigV2
  Pds::TypeId cfgTypeId2(Pds::TypeId::Id_CspadConfig, 2);
  if ( const Pds::CsPad::ConfigV2* config = m_configStore.find<Pds::CsPad::ConfigV2>(cfgTypeId2, src.top()) ) {
    qMask = config->quadMask();
    for (int q = 0 ; q != Pds::CsPad::MaxQuadsPerSensor; ++ q) {
      sMask[q] = config->roiMask(q);
      sections += ::bitCount(sMask[q], Pds::CsPad::ASICsPerQuad/2);
    }
  } else {
    MsgLog ( logger, error, "CsPadElementV2Cvt - no configuration object was defined" );
    return ;
  }

  // create all containers if running first time
  if ( not m_elementCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::CsPadElementV2> elContFactory( "element", m_chunk_size, m_deflate ) ;
    m_elementCont = new ElementCont ( elContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate ) ;
    m_pixelDataCont = new PixelDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }

  // get few constants
  const unsigned nQuad = ::bitCount(qMask, Pds::CsPad::MaxQuadsPerSensor);
  const unsigned nSect = sections;
  const unsigned ssize = Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

  // make data arrays
  H5DataTypes::CsPadElementV2 elems[nQuad] ;
  uint16_t pixelData[nSect][Pds::CsPad::ColumnsPerASIC][Pds::CsPad::MaxRowsPerASIC*2];

  // move the data
  const XtcType* pdselem = &data ;
  unsigned quad = 0;
  unsigned sect = 0;
  for ( unsigned iq = 0 ; iq != Pds::CsPad::MaxQuadsPerSensor ; ++ iq ) {
        
    if ( not (qMask & (1 << iq))) continue;

    // copy frame info
    elems[quad] = H5DataTypes::CsPadElementV2(*pdselem) ;

    // start of pixel data
    const uint16_t* sdata = (const uint16_t*)(pdselem+1);

    for ( unsigned is = 0 ; is != Pds::CsPad::ASICsPerQuad/2 ; ++ is ) {
    
      if ( not (sMask[iq] & (1 << is))) continue; 
  
      // copy pixel data
      std::copy(sdata, sdata+ssize, &pixelData[sect][0][0]);

      // advance to next section
      sdata += ssize;
      ++ sect;
    }

    // move to next frame
    pdselem = (const XtcType*)(sdata+2) ;
    ++ quad;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::CsPadElementV2::stored_type(nQuad);
  m_elementCont->container(group,type)->append ( elems[0], type ) ;
  type = H5DataTypes::CsPadElementV2::stored_data_type(nSect) ;
  m_pixelDataCont->container(group,type)->append ( pixelData[0][0][0], type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CsPadElementV2Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_elementCont ) m_elementCont->closeGroup( group ) ;
  if ( m_pixelDataCont ) m_pixelDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
