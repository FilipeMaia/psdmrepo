//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV1ToV2...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/TimepixDataV1ToV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_pds2psana {

//----------------
// Constructors --
//----------------
TimepixDataV1ToV2::TimepixDataV1ToV2(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Timepix::DataV2()
  , m_xtcObj(xtcPtr)
  , m_data(0)
{
}

//--------------
// Destructor --
//--------------
TimepixDataV1ToV2::~TimepixDataV1ToV2 ()
{
  delete [] m_data;
}

uint16_t
TimepixDataV1ToV2::width() const
{
  return m_xtcObj->width();
}

uint16_t
TimepixDataV1ToV2::height() const {
  return m_xtcObj->height();
}

uint32_t
TimepixDataV1ToV2::timestamp() const
{
  return m_xtcObj->timestamp();
}

uint16_t
TimepixDataV1ToV2::frameCounter() const
{
  return m_xtcObj->frameCounter();
}

uint16_t
TimepixDataV1ToV2::lostRows() const
{
  return m_xtcObj->lostRows();
}

ndarray<const uint16_t, 2>
TimepixDataV1ToV2::data() const
{
  uint16_t width = m_xtcObj->width();
  uint16_t height = m_xtcObj->height();

  if (not m_data) {

    // get DataV1 data
    ndarray<const uint16_t, 2> data1 = m_xtcObj->data();

    // allocate enough space
    m_data = new uint16_t[data1.size()];

    // this is stolen from pdsdata/timepix/src/DataV2.cc
    // convert data
    unsigned destX, destY;
    const uint16_t *src = data1.data();

    for (unsigned iy=0; iy < height * 2u; iy++) {
      for (unsigned ix=0; ix < width / 2u; ix++) {
        // map pixels from 256x1024 to 512x512
        switch (iy / 256) {
          case 0:
            destX = iy;
            destY = 511 - ix;
            break;
          case 1:
            destX = iy - 256;
            destY = 255 - ix;
            break;
          case 2:
            destX = 1023 - iy;
            destY = ix;
            break;
          case 3:
            destX = 1023 + 256 - iy;
            destY = ix + 256;
            break;
          default:
            // error
            destX = destY = 0;
            break;
        }
        m_data[(destY * width) + destX] = src[(iy * width / 2) + ix];
      }
    }

  }

  return make_ndarray(m_data, height, width);
}

uint32_t
TimepixDataV1ToV2::depth() const
{
  return m_xtcObj->depth();
}

uint32_t
TimepixDataV1ToV2::depth_bytes() const
{
  return m_xtcObj->depth_bytes();
}

uint32_t
TimepixDataV1ToV2::data_size() const
{
  return m_xtcObj->data_size();
}

} // namespace psddl_pds2psana
