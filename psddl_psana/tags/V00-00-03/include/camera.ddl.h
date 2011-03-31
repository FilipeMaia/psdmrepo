#ifndef PSANA_CAMERA_DDL_H
#define PSANA_CAMERA_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

namespace Psana {
namespace Camera {

/** Class: FrameCoord
  
*/


class FrameCoord {
public:
  FrameCoord()
  {
  }
  FrameCoord(uint16_t arg__column, uint16_t arg__row)
    : _column(arg__column), _row(arg__row)
  {
  }
  uint16_t column() const {return _column;}
  uint16_t row() const {return _row;}
  static uint32_t _sizeof()  {return 4;}
private:
  uint16_t	_column;
  uint16_t	_row;
};

/** Class: FrameFccdConfigV1
  
*/


class FrameFccdConfigV1 {
public:
  enum {Version = 1};
  enum {TypeId = Pds::TypeId::Id_FrameFccdConfig};
  virtual ~FrameFccdConfigV1();
};

/** Class: FrameFexConfigV1
  
*/


class FrameFexConfigV1 {
public:
  enum {Version = 1};
  enum {TypeId = Pds::TypeId::Id_FrameFexConfig};
  enum Forwarding {
    NoFrame,
    FullFrame,
    RegionOfInterest,
  };
  enum Processing {
    NoProcessing,
    GssFullFrame,
    GssRegionOfInterest,
    GssThreshold,
  };
  virtual ~FrameFexConfigV1();
  virtual Camera::FrameFexConfigV1::Forwarding forwarding() const = 0;
  virtual uint32_t forward_prescale() const = 0;
  virtual Camera::FrameFexConfigV1::Processing processing() const = 0;
  virtual const Camera::FrameCoord& roiBegin() const = 0;
  virtual const Camera::FrameCoord& roiEnd() const = 0;
  virtual uint32_t threshold() const = 0;
  virtual uint32_t number_of_masked_pixels() const = 0;
  virtual const Camera::FrameCoord& masked_pixel_coordinates(uint32_t i0) const = 0;
  virtual std::vector<int> masked_pixel_shape() const = 0;
};

/** Class: FrameV1
  
*/


class FrameV1 {
public:
  enum {Version = 1};
  enum {TypeId = Pds::TypeId::Id_Frame};
  virtual ~FrameV1();
  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;
  virtual uint32_t depth() const = 0;
  virtual uint32_t offset() const = 0;
  virtual const uint8_t* data() const = 0;
  virtual std::vector<int> data_shape() const = 0;
};

/** Class: TwoDGaussianV1
  
*/


class TwoDGaussianV1 {
public:
  enum {Version = 1};
  enum {TypeId = Pds::TypeId::Id_TwoDGaussian};
  virtual ~TwoDGaussianV1();
  virtual uint64_t integral() const = 0;
  virtual double xmean() const = 0;
  virtual double ymean() const = 0;
  virtual double major_axis_width() const = 0;
  virtual double minor_axis_width() const = 0;
  virtual double major_axis_tilt() const = 0;
};
} // namespace Camera
} // namespace Psana
#endif // PSANA_CAMERA_DDL_H
