#ifndef PSANA_OPAL1K_DDL_H
#define PSANA_OPAL1K_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

#include "psddl_psana/camera.ddl.h"
namespace Psana {
namespace Opal1k {

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_Opal1kConfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  enum {
    LUT_Size = 4096 /**<  */
  };
  enum {
    Row_Pixels = 1024 /**<  */
  };
  enum {
    Column_Pixels = 1024 /**<  */
  };
  enum {
    Output_LUT_Size = 4096 /**<  */
  };

  /** Image bit depth modes. */
  enum Depth {
    Eight_bit,
    Ten_bit,
    Twelve_bit,
  };
  enum Binning {
    x1,
    x2,
    x4,
    x8,
  };
  enum Mirroring {
    None,
    HFlip,
    VFlip,
    HVFlip,
  };
  virtual ~ConfigV1();
  /** offset/pedestal setting for camera (before gain) */
  virtual uint16_t black_level() const = 0;
  /** camera gain setting in percentile [100-3200] = [1x-32x] */
  virtual uint16_t gain_percent() const = 0;
  /** bit-depth of pixel counts */
  virtual Opal1k::ConfigV1::Depth output_resolution() const = 0;
  /** vertical re-binning of output (consecutive rows summed) */
  virtual Opal1k::ConfigV1::Binning vertical_binning() const = 0;
  /** geometric transformation of the image */
  virtual Opal1k::ConfigV1::Mirroring output_mirroring() const = 0;
  /** 1: remap the pixels to appear in natural geometric order 
	                  (left->right, top->bottom);
	            0: pixels appear on dual taps from different rows
	                  (left->right, top->bottom) alternated with
	                  (left->right, bottom->top) pixel by pixel */
  virtual uint8_t vertical_remapping() const = 0;
  /** correct defective pixels internally */
  virtual uint8_t defect_pixel_correction_enabled() const = 0;
  /** apply output lookup table corrections */
  virtual uint8_t output_lookup_table_enabled() const = 0;
  virtual uint32_t number_of_defect_pixels() const = 0;
  virtual const uint16_t* output_lookup_table() const = 0;
  virtual const Camera::FrameCoord& defect_pixel_coordinates(uint32_t i0) const = 0;
  /** offset/pedestal value in pixel counts */
  virtual uint16_t output_offset() const = 0;
  /** bit-depth of pixel counts */
  virtual uint32_t output_resolution_bits() const = 0;
  /** Method which returns the shape (dimensions) of the data returned by output_lookup_table() method. */
  virtual std::vector<int> output_lookup_table_shape() const = 0;
  /** Method which returns the shape (dimensions) of the data returned by defect_pixel_coordinates() method. */
  virtual std::vector<int> defect_pixel_coordinates_shape() const = 0;
};
} // namespace Opal1k
} // namespace Psana
#endif // PSANA_OPAL1K_DDL_H
