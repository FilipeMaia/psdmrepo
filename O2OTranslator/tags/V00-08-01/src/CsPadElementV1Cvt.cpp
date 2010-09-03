//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadElementV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadElementV1Cvt.h"

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

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  const char logger[] = "CsPadElementV1Cvt" ;
  
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
CsPadElementV1Cvt::CsPadElementV1Cvt ( const std::string& typeGroupName,
                                   const ConfigObjectStore& configStore,
                                   hsize_t chunk_size,
                                   int deflate )
  : EvtDataTypeCvt<Pds::CsPad::ElementV1>(typeGroupName)
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
CsPadElementV1Cvt::~CsPadElementV1Cvt ()
{
  delete m_elementCont ;
  delete m_pixelDataCont ;
  delete m_timeCont ;
}

// typed conversion method
void
CsPadElementV1Cvt::typedConvertSubgroup ( hdf5pp::Group group,
                                          const XtcType& data,
                                          size_t size,
                                          const Pds::TypeId& typeId,
                                          const O2OXtcSrc& src,
                                          const H5DataTypes::XtcClockTime& time )
{
  // find corresponding configuration object
  Pds::TypeId cfgTypeId(Pds::TypeId::Id_CspadConfig, 1);
  const Pds::CsPad::ConfigV1* config = m_configStore.find<Pds::CsPad::ConfigV1>(cfgTypeId, src.top());
  if ( not config ) {
    MsgLog ( logger, error, "CsPadElementV1Cvt - no configuration object was defined" );
    return ;
  }

  // create all containers if running first time
  if ( not m_elementCont ) {

    // create container for frames
    CvtDataContFactoryTyped<H5DataTypes::CsPadElementV1> elContFactory( "element", m_chunk_size, m_deflate ) ;
    m_elementCont = new ElementCont ( elContFactory ) ;

    // create container for frame data
    CvtDataContFactoryTyped<uint16_t> dataContFactory( "data", m_chunk_size, m_deflate ) ;
    m_pixelDataCont = new PixelDataCont ( dataContFactory ) ;

    // make container for time
    CvtDataContFactoryDef<H5DataTypes::XtcClockTime> timeContFactory ( "time", m_chunk_size, m_deflate ) ;
    m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;

  }


  // get few constants
  const unsigned nElem = ::bitCount(config->quadMask(), Pds::CsPad::MaxQuadsPerSensor);
  const unsigned nAsic = config->numAsicsRead();

  // make data arrays
  H5DataTypes::CsPadElementV1 elems[nElem] ;
  uint16_t pixelData[nElem][nAsic][Pds::CsPad::MaxRowsPerASIC][Pds::CsPad::ColumnsPerASIC];

  // move the data
  const XtcType* pdselem = &data ;
  for ( unsigned iElem = 0 ; iElem != nElem ; ++ iElem ) {

    // copy frame info
    elems[iElem] = H5DataTypes::CsPadElementV1(*pdselem) ;

    // copy data
    for ( unsigned iAsic = 0 ; iAsic != nAsic ; ++ iAsic ) {
      for ( unsigned row = 0 ; row < Pds::CsPad::MaxRowsPerASIC ; ++ row ) {
        for ( unsigned col = 0 ; col != Pds::CsPad::ColumnsPerASIC ; ++ col ) {
          pixelData[iElem][iAsic][row][col] = *pdselem->pixel(iAsic, col, row);
        }
      }
    }

    // move to next frame
    pdselem = pdselem->next(*config) ;
  }

  // store the data
  hdf5pp::Type type = H5DataTypes::CsPadElementV1::stored_type( *config );
  m_elementCont->container(group,type)->append ( elems[0], type ) ;
  type = H5DataTypes::CsPadElementV1::stored_data_type ( *config ) ;
  m_pixelDataCont->container(group,type)->append ( pixelData[0][0][0][0], type ) ;
  m_timeCont->container(group)->append ( time ) ;
}

/// method called when the driver closes a group in the file
void
CsPadElementV1Cvt::closeSubgroup( hdf5pp::Group group )
{
  if ( m_elementCont ) m_elementCont->closeGroup( group ) ;
  if ( m_pixelDataCont ) m_pixelDataCont->closeGroup( group ) ;
  if ( m_timeCont ) m_timeCont->closeGroup( group ) ;
}

} // namespace O2OTranslator
