#ifndef PSANA_PNCCD_DDL_H
#define PSANA_PNCCD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "pdsdata/xtc/TypeId.hh"

#include <vector>

namespace Psana {
namespace PNCCD {

/** @class ConfigV1

  pnCCD configuration class ConfigV1
*/


class ConfigV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_pnCCDconfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  virtual ~ConfigV1();
  /** Number of links in the pnCCD. */
  virtual uint32_t numLinks() const = 0;
  /** Size of the payload in bytes for single link */
  virtual uint32_t payloadSizePerLink() const = 0;
};

/** @class ConfigV2

  pnCCD configuration class ConfigV2
*/


class ConfigV2 {
public:
  enum {
    Version = 2 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_pnCCDconfig /**< XTC type ID value (from Pds::TypeId class) */
  };
  virtual ~ConfigV2();
  /** Number of links in the pnCCD. */
  virtual uint32_t numLinks() const = 0;
  /** Size of the payload in bytes for single link */
  virtual uint32_t payloadSizePerLink() const = 0;
  /** Number of channels */
  virtual uint32_t numChannels() const = 0;
  /** Number of rows */
  virtual uint32_t numRows() const = 0;
  /** Number of submodule channels */
  virtual uint32_t numSubmoduleChannels() const = 0;
  /** Number of submodule rows */
  virtual uint32_t numSubmoduleRows() const = 0;
  /** Number of submodules */
  virtual uint32_t numSubmodules() const = 0;
  /** Magic word from CAMEX */
  virtual uint32_t camexMagic() const = 0;
  /** Information data string */
  virtual const char* info() const = 0;
  /** Timing file name string */
  virtual const char* timingFName() const = 0;
  /** Method which returns the shape (dimensions) of the data returned by info() method. */
  virtual std::vector<int> info_shape() const = 0;
  /** Method which returns the shape (dimensions) of the data returned by timingFName() method. */
  virtual std::vector<int> timingFName_shape() const = 0;
};

/** @class FrameV1

  pnCCD configuration class FrameV1
*/

class ConfigV1;
class ConfigV2;

class FrameV1 {
public:
  enum {
    Version = 1 /**< XTC type version number */
  };
  enum {
    TypeId = Pds::TypeId::Id_pnCCDframe /**< XTC type ID value (from Pds::TypeId class) */
  };
  virtual ~FrameV1();
  /** Special values */
  virtual uint32_t specialWord() const = 0;
  /** Frame number */
  virtual uint32_t frameNumber() const = 0;
  /** Most significant part of timestamp */
  virtual uint32_t timeStampHi() const = 0;
  /** Least significant part of timestamp */
  virtual uint32_t timeStampLo() const = 0;
  /** Frame data */
  virtual const uint16_t* data() const = 0;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  virtual std::vector<int> data_shape() const = 0;
};
} // namespace PNCCD
} // namespace Psana
#endif // PSANA_PNCCD_DDL_H
