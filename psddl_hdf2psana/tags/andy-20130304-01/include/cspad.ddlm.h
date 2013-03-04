#ifndef PSDDL_HDF2PSANA_CSPAD_DDLM_H
#define PSDDL_HDF2PSANA_CSPAD_DDLM_H 1

#include "psddl_hdf2psana/cspad.ddl.h"


namespace psddl_hdf2psana {
namespace CsPad {

template <typename Config>
class DataV1_v0 : public Psana::CsPad::DataV1 {
public:
  typedef Psana::CsPad::DataV1 PsanaType;
  DataV1_v0() {}
  DataV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV1_v0() {}

  /** Data objects, one element per quadrant. The size of the array is determined by
            the numQuads() method of the configuration object. */
  virtual const Psana::CsPad::ElementV1& quads(uint32_t i0) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  virtual std::vector<int> quads_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

};

template <typename Config>
class DataV2_v0 : public Psana::CsPad::DataV2 {
public:
  typedef Psana::CsPad::DataV1 PsanaType;
  DataV2_v0() {}
  DataV2_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV2_v0() {}

  /** Data objects, one element per quadrant. The size of the array is determined by
            the numQuads() method of the configuration object. */
  virtual const Psana::CsPad::ElementV2& quads(uint32_t i0) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  virtual std::vector<int> quads_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

};

} // namespace CsPad
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CSPAD_DDLM_H
