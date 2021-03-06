#ifndef PSANA_GENERICPGP_DDL_H
#define PSANA_GENERICPGP_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <iosfwd>
#include <cstring>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
namespace Psana {
namespace GenericPgp {

/** @class CDimension

  
*/


class CDimension {
public:
  virtual ~CDimension();
  virtual uint32_t rows() const = 0;
  virtual uint32_t columns() const = 0;
};

/** @class CRegister

  
*/


class CRegister {
public:
  enum Action {
    RegisterRead = 0, /**< Read and store */
    RegisterWrite = 1, /**< Write */
    RegisterWriteA = 2, /**< Write and wait for ack */
    RegisterVerify = 3, /**< Read and verify */
    Spin = 4, /**< Spin lock */
    Usleep = 5, /**< Sleep */
    Flush = 6, /**< Flush pending reads */
  };
  CRegister(GenericPgp::CRegister::Action arg__action, uint32_t arg__datasize, uint32_t arg__address, uint32_t arg__offset, uint32_t arg__mask)
    : _Action(((arg__action) & 0xff)|(((arg__datasize) & 0xffffff)<<8)), _address(arg__address), _offset(arg__offset), _mask(arg__mask)
  {
  }
  CRegister() {}
  /** Configuration action */
  GenericPgp::CRegister::Action action() const { return Action(this->_Action & 0xff); }
  /** Size of register access (in uint32_t's) */
  uint32_t datasize() const { return uint32_t((this->_Action>>8) & 0xffffff); }
  /** Register access address */
  uint32_t address() const { return _address; }
  /** Payload offset */
  uint32_t offset() const { return _offset; }
  /** Register value mask */
  uint32_t mask() const { return _mask; }
  static uint32_t _sizeof() { return 16; }
private:
  uint32_t	_Action;
  uint32_t	_address;	/**< Register access address */
  uint32_t	_offset;	/**< Payload offset */
  uint32_t	_mask;	/**< Register value mask */
};
std::ostream& operator<<(std::ostream& str, GenericPgp::CRegister::Action enval);

/** @class CStream

  
*/


class CStream {
public:
  CStream(uint32_t arg__pgp_channel, uint32_t arg__data_type, uint32_t arg__config_type, uint32_t arg__config_offset)
    : _pgp_channel(arg__pgp_channel), _data_type(arg__data_type), _config_type(arg__config_type), _config_offset(arg__config_offset)
  {
  }
  CStream() {}
  /** PGP virtual channel ID */
  uint32_t pgp_channel() const { return _pgp_channel; }
  /** Event data type ID */
  uint32_t data_type() const { return _data_type; }
  /** Configuration data type ID */
  uint32_t config_type() const { return _config_type; }
  /** Location of configuration data */
  uint32_t config_offset() const { return _config_offset; }
  static uint32_t _sizeof() { return 16; }
private:
  uint32_t	_pgp_channel;	/**< PGP virtual channel ID */
  uint32_t	_data_type;	/**< Event data type ID */
  uint32_t	_config_type;	/**< Configuration data type ID */
  uint32_t	_config_offset;	/**< Location of configuration data */
};

/** @class ConfigV1

  
*/


class ConfigV1 {
public:
  enum { TypeId = Pds::TypeId::Id_GenericPgpConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  virtual ~ConfigV1();
  /** Serial number identifying the array */
  virtual uint32_t id() const = 0;
  /** Dimensions of the frame data from the array */
  virtual const GenericPgp::CDimension& frame_dim() const = 0;
  /** Dimensions of the auxillary data from the array */
  virtual const GenericPgp::CDimension& aux_dim() const = 0;
  /** Dimensions of the environmental data from the array */
  virtual const GenericPgp::CDimension& env_dim() const = 0;
  /** Number of registers in the sequence array */
  virtual uint32_t number_of_registers() const = 0;
  /** Number of (sub)sequences of register operations in the array */
  virtual uint32_t number_of_sequences() const = 0;
  virtual uint32_t number_of_streams() const = 0;
  virtual uint32_t payload_size() const = 0;
  virtual ndarray<const uint32_t, 2> pixel_settings() const = 0;
  /** Lengths of (sub)sequence register operations in the array */
  virtual ndarray<const uint32_t, 1> sequence_length() const = 0;
  /** Register Operations */
  virtual ndarray<const GenericPgp::CRegister, 1> sequence() const = 0;
  /** Stream readout configuration */
  virtual ndarray<const GenericPgp::CStream, 1> stream() const = 0;
  /** Stream and Register Data */
  virtual ndarray<const uint32_t, 1> payload() const = 0;
  /** Number of rows in a readout unit */
  virtual uint32_t numberOfRows() const = 0;
  /** Number of columns in a readout unit */
  virtual uint32_t numberOfColumns() const = 0;
  /** Number of rows in the auxillary data */
  virtual uint32_t lastRowExclusions() const = 0;
  /** Number of elements in environmental data */
  virtual uint32_t numberOfAsics() const = 0;
};
} // namespace GenericPgp
} // namespace Psana
#endif // PSANA_GENERICPGP_DDL_H
