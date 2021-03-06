#ifndef PSANA_PARTITION_DDL_H
#define PSANA_PARTITION_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <iosfwd>
#include <cstring>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/Src.hh"
namespace Psana {
namespace Partition {

/** @class Source

  
*/


class Source {
public:
  Source(const Pds::Src& arg__src, uint32_t arg__group);
  Source() {}
  const Pds::Src& src() const { return _src; }
  uint32_t group() const { return _group; }
  static uint32_t _sizeof() { return (((((0+(Pds::Src::_sizeof()))+4)+4)-1)/4)*4; }
private:
  Pds::Src	_src;
  uint32_t	_group;
};

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_PartitionConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  virtual ~ConfigV1();
  /** Mask of requested BLD */
  virtual uint64_t bldMask() const = 0;
  /** Number of source definitions */
  virtual uint32_t numSources() const = 0;
  /** Source configuration objects */
  virtual ndarray<const Partition::Source, 1> sources() const = 0;
};
} // namespace Partition
} // namespace Psana
#endif // PSANA_PARTITION_DDL_H
