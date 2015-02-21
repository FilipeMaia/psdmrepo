@include "psddldata/camera.ddl";
@package Quartz  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_QuartzConfig, 1)]]
  [[config_type]]
{
  @const int32_t LUT_Size = 4096;
  @const int32_t Row_Pixels = 2048;
  @const int32_t Column_Pixels = 2048;
  @const int32_t Output_LUT_Size = 4096;

  /* Image bit depth modes. */
  @enum Depth (uint8_t) {
    Eight_bit,
    Ten_bit,
  }
  @enum Binning (uint8_t) {
    x1,
    x2,
    x4,
  }
  @enum Mirroring (uint8_t) {
    None,
    HFlip,
    VFlip,
    HVFlip,
  }

  uint32_t _offsetAndGain {	/* offset and gain */
    uint16_t _bf_offset:16 -> black_level;	/* offset/pedestal setting for camera (before gain) */
    uint16_t _bf_gain:16 -> gain_percent;	/* camera gain setting in percentile [100-3200] = [1x-32x] */
  }
  uint32_t _outputOptions {	/* bit mask of output formatting options */
    Depth _bf_resol:4 -> output_resolution;	/* bit-depth of pixel counts */
    Binning _bf_hbin:2 -> horizontal_binning;	/* horizontal re-binning of output (consecutive rows summed) */
    Binning _bf_vbin:2 -> vertical_binning;	/* vertical re-binning of output (consecutive rows summed) */
    Mirroring _bf_mirr:4 -> output_mirroring;	/* geometric transformation of the image */
    uint8_t _bf_lookup:1 -> output_lookup_table_enabled;	/* apply output lookup table corrections */
    uint8_t _bf_corr:1 -> defect_pixel_correction_enabled;	/* correct defective pixels internally */
  }
  uint32_t _defectPixelCount -> number_of_defect_pixels;
  uint16_t _lookup_table[Output_LUT_Size*@self.output_lookup_table_enabled()] -> output_lookup_table;
  Camera.FrameCoord _defectPixels[@self.number_of_defect_pixels()] -> defect_pixel_coordinates;

  /* offset/pedestal value in pixel counts */
  uint16_t output_offset()
  [[language("C++")]] @{ return (uint32_t(@self.black_level())*uint32_t(@self.gain_percent()))/100; @}

  /* bit-depth of pixel counts */
  uint32_t output_resolution_bits()
  [[language("C++")]] @{ return @self.output_resolution()*2+8; @}

  /* Standard constructor. */
  @init()  [[auto, inline]];

}


//------------------ ConfigV2 ------------------
@type ConfigV2
  [[type_id(Id_QuartzConfig, 2)]]
  [[config_type]]
{
  @const int32_t LUT_Size = 4096;
  @const int32_t Row_Pixels = 2048;
  @const int32_t Column_Pixels = 2048;
  @const int32_t Output_LUT_Size = 4096;

  /* Image bit depth modes. */
  @enum Depth (uint8_t) {
    Eight_bit,
    Ten_bit,
  }
  @enum Binning (uint8_t) {
    x1,
    x2,
    x4,
  }
  @enum Mirroring (uint8_t) {
    None,
    HFlip,
    VFlip,
    HVFlip,
  }

  uint32_t _offsetAndGain {	/* offset and gain */
    uint16_t _bf_offset:16 -> black_level;	/* offset/pedestal setting for camera (before gain) */
    uint16_t _bf_gain:16 -> gain_percent;	/* camera gain setting in percentile [100-3200] = [1x-32x] */
  }
  uint32_t _outputOptions {	/* bit mask of output formatting options */
    Depth _bf_resol:4 -> output_resolution;	/* bit-depth of pixel counts */
    Binning _bf_hbin:2 -> horizontal_binning;	/* horizontal re-binning of output (consecutive rows summed) */
    Binning _bf_vbin:2 -> vertical_binning;	/* vertical re-binning of output (consecutive rows summed) */
    Mirroring _bf_mirr:4 -> output_mirroring;	/* geometric transformation of the image */
    uint8_t _bf_lookup:1 -> output_lookup_table_enabled;	/* apply output lookup table corrections */
    uint8_t _bf_corr:1 -> defect_pixel_correction_enabled;	/* correct defective pixels internally */
    uint8_t _use_roi:1 -> use_hardware_roi;     /* enable hardware region of interest */
    uint8_t _use_test_pattern:1 -> use_test_pattern;     /* enable the test pattern */
    uint8_t _max_taps:4 -> max_taps;            /* maximum taps in output driver */
  }

  /* hardware ROI begin */
  Camera.FrameCoord _roi_lo -> roi_lo;

  /* hardware ROI end */
  Camera.FrameCoord _roi_hi -> roi_hi;

  uint32_t _defectPixelCount -> number_of_defect_pixels;
  uint16_t _lookup_table[Output_LUT_Size*@self.output_lookup_table_enabled()] -> output_lookup_table;
  Camera.FrameCoord _defectPixels[@self.number_of_defect_pixels()] -> defect_pixel_coordinates;

  /* offset/pedestal value in pixel counts */
  uint16_t output_offset()
  [[language("C++")]] @{ return (uint32_t(@self.black_level())*uint32_t(@self.gain_percent()))/100; @}

  /* bit-depth of pixel counts */
  uint32_t output_resolution_bits()
  [[language("C++")]] @{ return @self.output_resolution()*2+8; @}

  /* Standard constructor. */
  @init()  [[auto, inline]];

}
} //- @package Quartz
