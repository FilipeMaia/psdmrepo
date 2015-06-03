//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdFullFrameV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/PnccdFullFrameV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>

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
PnccdFullFrameV1::PnccdFullFrameV1 (const Psana::PNCCD::FramesV1& frames)
  : FullFrameV1()
  , _specialWord(0)
  , _frameNumber(0)
  , _timeStampHi(0)
  , _timeStampLo(0)
{
  assert(frames.numLinks() == 4);

  // copy few items from first frame, other frames should heve identical values
  const Psana::PNCCD::FrameV1& frame0 = frames.frame(0);
  _specialWord = frame0.specialWord();
  _frameNumber = frame0.frameNumber();

  // take the lowest timestamp of four frames
  for (unsigned i = 0; i != 4; ++ i) {
    const Psana::PNCCD::FrameV1& frame = frames.frame(i);
    uint32_t tshi = frame.timeStampHi(), tslo = frame.timeStampLo();
    if (i == 0 or tshi < _timeStampHi or (tshi == _timeStampHi and tslo < _timeStampLo)) {
      _timeStampHi = tshi;
      _timeStampLo = tslo;
    }
  }

  // make large image out of four small images
  uint16_t* dest = &_data[0][0];
  ndarray<const uint16_t, 2>::iterator src0 = frames.frame(0).data().begin();
  ndarray<const uint16_t, 2>::iterator src3 = frames.frame(3).data().begin();
  for (int iY = 0; iY < 512; ++ iY, src0 += 512, src3 += 512) {
    dest = std::copy(src0, src0+512, dest);
    dest = std::copy(src3, src3+512, dest);
  }

  ndarray<const uint16_t, 2>::reverse_iterator src1 = frames.frame(1).data().rbegin();
  ndarray<const uint16_t, 2>::reverse_iterator src2 = frames.frame(2).data().rbegin();
  for (int iY = 0; iY < 512; ++ iY, src1 += 512, src2 += 512) {
    dest = std::copy(src1, src1+512, dest);
    dest = std::copy(src2, src2+512, dest);
  }
}

//--------------
// Destructor --
//--------------
PnccdFullFrameV1::~PnccdFullFrameV1 ()
{
}

/** Special values */
uint32_t
PnccdFullFrameV1::specialWord() const
{
  return _specialWord;
}

/** Frame number */
uint32_t
PnccdFullFrameV1::frameNumber() const
{
  return _frameNumber;
}

/** Most significant part of timestamp */
uint32_t
PnccdFullFrameV1::timeStampHi() const
{
  return _timeStampHi;
}

/** Least significant part of timestamp */
uint32_t
PnccdFullFrameV1::timeStampLo() const
{
  return _timeStampLo;
}

/** Full frame data, image size is 1024x1024. */
ndarray<const uint16_t, 2>
PnccdFullFrameV1::data() const
{
  return make_ndarray(&_data[0][0], 1024, 1024);
}


} // namespace psddl_pds2psana
