@package Camera  {


//------------------ FrameCoord ------------------
/* Class representing the coordinates of pixels inside the camera frame. */
@type FrameCoord
  [[value_type]]
{
  uint16_t _column -> column;	/* Column index (x value). */
  uint16_t _row -> row;	/* Row index (y value). */

  /* Constructor which takes values for every attribute */
  @init()  [[auto, inline]];

}


//------------------ FrameFccdConfigV1 ------------------
/* This class was never defined/implemented. */
@type FrameFccdConfigV1
  [[type_id(Id_FrameFccdConfig, 1)]]
  [[config_type]]
{
}


//------------------ FrameFexConfigV1 ------------------
/* Class containing configuration data for online frame feature extraction process. */
@type FrameFexConfigV1
  [[type_id(Id_FrameFexConfig, 1)]]
  [[config_type]]
{
  @enum Forwarding (uint32_t) {
    NoFrame,
    FullFrame,
    RegionOfInterest,
  }
  @enum Processing (uint32_t) {
    NoProcessing,
    GssFullFrame,
    GssRegionOfInterest,
    GssThreshold,
  }

  Forwarding _forwarding -> forwarding;	/* frame forwarding policy */
  uint32_t _forward_prescale -> forward_prescale;	/* Prescale of events with forwarded frames */
  Processing _processing -> processing;	/* algorithm to apply to frames to produce processed output */
  FrameCoord _roiBegin -> roiBegin;	/* Coordinate of start of rectangular region of interest (inclusive). */
  FrameCoord _roiEnd -> roiEnd;	/* Coordinate of finish of rectangular region of interest (exclusive). */
  uint32_t _threshold -> threshold;	/* Pixel data threshold value to apply in processing. */
  uint32_t _masked_pixel_count -> number_of_masked_pixels;	/* Count of masked pixels to exclude from processing. */
  FrameCoord _masked_pixel_coordinates[@self.number_of_masked_pixels()] -> masked_pixel_coordinates  [[shape_method(masked_pixel_shape)]];	/* Location of masked pixel coordinates. */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}


//------------------ FrameV1 ------------------
@type FrameV1
  [[type_id(Id_Frame, 1)]]
{
  uint32_t _width -> width;	/* Number of pixels in a row. */
  uint32_t _height -> height;	/* Number of pixels in a column. */
  uint32_t _depth -> depth;	/* Number of bits per pixel. */
  uint32_t _offset -> offset;	/* Fixed offset/pedestal value of pixel data. */
  uint8_t _pixel_data[@self._width*@self._height*((@self._depth+7)/8)] -> _int_pixel_data;	/* Pixel data as array of bytes, method is for internal use only, use data8() or 
            data16() for access to the data. */

  /* Returns pixel data array when stored data type is 8-bit (depth() is less than 9).
                If data type is 16-bit then empty array is returned, use data16() method in this case. */
  uint8_t[][] data8()
  [[language("C++")]] @{
    if (@self.depth() > 8) return ndarray<const uint8_t, 2>(); 
    return make_ndarray(_int_pixel_data().data(), height(), width());
  @}

  /* Returns pixel data array when stored data type is 16-bit (depth() is greater than 8).
                If data type is 8-bit then empty array is returned, use data8() method in this case. */
  uint16_t[][] data16()
  [[language("C++")]] @{
    if (@self.depth() <= 8) return ndarray<const uint16_t, 2>(); 
    return make_ndarray((const uint16_t*)_int_pixel_data().data(), height(), width());
  @}

  /* Number of bytes per pixel. */
  uint32_t depth_bytes()  [[inline]]
  [[language("C++")]] @{ return (@self.depth()+7)/8; @}

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Construct from dimensions only.  Allow data to be appended externally. */
  @init(width -> _width, height -> _height, depth -> _depth, offset -> _offset)  [[inline]];

}


//------------------ TwoDGaussianV1 ------------------
@type TwoDGaussianV1
  [[type_id(Id_TwoDGaussian, 1)]]
{
  uint64_t _integral -> integral;
  double _xmean -> xmean;
  double _ymean -> ymean;
  double _major_axis_width -> major_axis_width;
  double _minor_axis_width -> minor_axis_width;
  double _major_axis_tilt -> major_axis_tilt;

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package Camera
