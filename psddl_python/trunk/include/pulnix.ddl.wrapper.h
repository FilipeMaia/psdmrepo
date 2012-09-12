/* Do not edit this file, as it is auto-generated */

#ifndef PSDDL_PYTHON_PULNIX_DDL_WRAPPER_H
#define PSDDL_PYTHON_PULNIX_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

namespace psddl_python {
namespace Pulnix {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class TM6740ConfigV1_Wrapper {
  shared_ptr<Psana::Pulnix::TM6740ConfigV1> _o;
  Psana::Pulnix::TM6740ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_TM6740Config };
  enum { Version = 1 };
  TM6740ConfigV1_Wrapper(shared_ptr<Psana::Pulnix::TM6740ConfigV1> obj) : _o(obj), o(_o.get()) {}
  TM6740ConfigV1_Wrapper(Psana::Pulnix::TM6740ConfigV1* obj) : o(obj) {}
  uint16_t gain_a() const { return o->gain_a(); }
  uint16_t gain_b() const { return o->gain_b(); }
  uint16_t vref() const { return o->vref(); }
  uint16_t shutter_width() const { return o->shutter_width(); }
  uint8_t gain_balance() const { return o->gain_balance(); }
  Psana::Pulnix::TM6740ConfigV1::Depth output_resolution() const { return o->output_resolution(); }
  Psana::Pulnix::TM6740ConfigV1::Binning horizontal_binning() const { return o->horizontal_binning(); }
  Psana::Pulnix::TM6740ConfigV1::Binning vertical_binning() const { return o->vertical_binning(); }
  Psana::Pulnix::TM6740ConfigV1::LookupTable lookuptable_mode() const { return o->lookuptable_mode(); }
  uint8_t output_resolution_bits() const { return o->output_resolution_bits(); }
};

class TM6740ConfigV2_Wrapper {
  shared_ptr<Psana::Pulnix::TM6740ConfigV2> _o;
  Psana::Pulnix::TM6740ConfigV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_TM6740Config };
  enum { Version = 2 };
  TM6740ConfigV2_Wrapper(shared_ptr<Psana::Pulnix::TM6740ConfigV2> obj) : _o(obj), o(_o.get()) {}
  TM6740ConfigV2_Wrapper(Psana::Pulnix::TM6740ConfigV2* obj) : o(obj) {}
  uint16_t gain_a() const { return o->gain_a(); }
  uint16_t gain_b() const { return o->gain_b(); }
  uint16_t vref_a() const { return o->vref_a(); }
  uint16_t vref_b() const { return o->vref_b(); }
  uint8_t gain_balance() const { return o->gain_balance(); }
  Psana::Pulnix::TM6740ConfigV2::Depth output_resolution() const { return o->output_resolution(); }
  Psana::Pulnix::TM6740ConfigV2::Binning horizontal_binning() const { return o->horizontal_binning(); }
  Psana::Pulnix::TM6740ConfigV2::Binning vertical_binning() const { return o->vertical_binning(); }
  Psana::Pulnix::TM6740ConfigV2::LookupTable lookuptable_mode() const { return o->lookuptable_mode(); }
  uint8_t output_resolution_bits() const { return o->output_resolution_bits(); }
};

  class TM6740ConfigV1_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Pulnix::TM6740ConfigV1";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Pulnix::TM6740ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Pulnix::TM6740ConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(TM6740ConfigV1_Wrapper(result)) : object();
    }
  };

  class TM6740ConfigV2_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Pulnix::TM6740ConfigV2";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Pulnix::TM6740ConfigV2::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Pulnix::TM6740ConfigV2> result = store.get(source, foundSrc);
      return result.get() ? object(TM6740ConfigV2_Wrapper(result)) : object();
    }
  };
} // namespace Pulnix
} // namespace psddl_python
#endif // PSDDL_PYTHON_PULNIX_DDL_WRAPPER_H
