/* Do not edit this file, as it is auto-generated */

#ifndef PSDDL_PYTHON_CAMERA_DDL_WRAPPER_H
#define PSDDL_PYTHON_CAMERA_DDL_WRAPPER_H 1

#include <psddl_python/DdlWrapper.h>
#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_psana/camera.ddl.h> // inc_psana

namespace psddl_python {
namespace Camera {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class FrameCoord_Wrapper {
  shared_ptr<Psana::Camera::FrameCoord> _o;
  Psana::Camera::FrameCoord* o;
public:
  FrameCoord_Wrapper(shared_ptr<Psana::Camera::FrameCoord> obj) : _o(obj), o(_o.get()) {}
  FrameCoord_Wrapper(Psana::Camera::FrameCoord* obj) : o(obj) {}
  uint16_t column() const { return o->column(); }
  uint16_t row() const { return o->row(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class FrameFccdConfigV1_Wrapper {
  shared_ptr<Psana::Camera::FrameFccdConfigV1> _o;
  Psana::Camera::FrameFccdConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_FrameFccdConfig };
  enum { Version = 1 };
  FrameFccdConfigV1_Wrapper(shared_ptr<Psana::Camera::FrameFccdConfigV1> obj) : _o(obj), o(_o.get()) {}
  FrameFccdConfigV1_Wrapper(Psana::Camera::FrameFccdConfigV1* obj) : o(obj) {}
};

class FrameFexConfigV1_Wrapper {
  shared_ptr<Psana::Camera::FrameFexConfigV1> _o;
  Psana::Camera::FrameFexConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_FrameFexConfig };
  enum { Version = 1 };
  FrameFexConfigV1_Wrapper(shared_ptr<Psana::Camera::FrameFexConfigV1> obj) : _o(obj), o(_o.get()) {}
  FrameFexConfigV1_Wrapper(Psana::Camera::FrameFexConfigV1* obj) : o(obj) {}
  Psana::Camera::FrameFexConfigV1::Forwarding forwarding() const { return o->forwarding(); }
  uint32_t forward_prescale() const { return o->forward_prescale(); }
  Psana::Camera::FrameFexConfigV1::Processing processing() const { return o->processing(); }
  const FrameCoord_Wrapper roiBegin() const { return FrameCoord_Wrapper(const_cast<Psana::Camera::FrameCoord*>(&o->roiBegin())); }
  const FrameCoord_Wrapper roiEnd() const { return FrameCoord_Wrapper(const_cast<Psana::Camera::FrameCoord*>(&o->roiEnd())); }
  uint32_t threshold() const { return o->threshold(); }
  uint32_t number_of_masked_pixels() const { return o->number_of_masked_pixels(); }
  vector<Psana::Camera::FrameCoord> masked_pixel_coordinates() const { VEC_CONVERT(o->masked_pixel_coordinates(), Psana::Camera::FrameCoord); }
};

class FrameV1_Wrapper {
  shared_ptr<Psana::Camera::FrameV1> _o;
  Psana::Camera::FrameV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_Frame };
  enum { Version = 1 };
  FrameV1_Wrapper(shared_ptr<Psana::Camera::FrameV1> obj) : _o(obj), o(_o.get()) {}
  FrameV1_Wrapper(Psana::Camera::FrameV1* obj) : o(obj) {}
  uint32_t width() const { return o->width(); }
  uint32_t height() const { return o->height(); }
  uint32_t depth() const { return o->depth(); }
  uint32_t offset() const { return o->offset(); }
  PyObject* _int_pixel_data() const { ND_CONVERT(o->_int_pixel_data(), uint8_t, 1); }
  PyObject* data8() const { ND_CONVERT(o->data8(), uint8_t, 2); }
  PyObject* data16() const { ND_CONVERT(o->data16(), uint16_t, 2); }
};

class TwoDGaussianV1_Wrapper {
  shared_ptr<Psana::Camera::TwoDGaussianV1> _o;
  Psana::Camera::TwoDGaussianV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_TwoDGaussian };
  enum { Version = 1 };
  TwoDGaussianV1_Wrapper(shared_ptr<Psana::Camera::TwoDGaussianV1> obj) : _o(obj), o(_o.get()) {}
  TwoDGaussianV1_Wrapper(Psana::Camera::TwoDGaussianV1* obj) : o(obj) {}
  uint64_t integral() const { return o->integral(); }
  double xmean() const { return o->xmean(); }
  double ymean() const { return o->ymean(); }
  double major_axis_width() const { return o->major_axis_width(); }
  double minor_axis_width() const { return o->minor_axis_width(); }
  double major_axis_tilt() const { return o->major_axis_tilt(); }
};

  class FrameCoord_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Camera::FrameCoord";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::Camera::FrameCoord> result = evt.get(source, key, foundSrc);
      return result.get() ? object(FrameCoord_Wrapper(result)) : object();
    }
  };

  class FrameFccdConfigV1_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Camera::FrameFccdConfigV1";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Camera::FrameFccdConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Camera::FrameFccdConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(FrameFccdConfigV1_Wrapper(result)) : object();
    }
  };

  class FrameFexConfigV1_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Camera::FrameFexConfigV1";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Camera::FrameFexConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Camera::FrameFexConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(FrameFexConfigV1_Wrapper(result)) : object();
    }
  };

  class FrameV1_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Camera::FrameV1";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    int getVersion() {
      return Psana::Camera::FrameV1::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::Camera::FrameV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(FrameV1_Wrapper(result)) : object();
    }
  };

  class TwoDGaussianV1_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Camera::TwoDGaussianV1";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    int getVersion() {
      return Psana::Camera::TwoDGaussianV1::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::Camera::TwoDGaussianV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TwoDGaussianV1_Wrapper(result)) : object();
    }
  };
} // namespace Camera
} // namespace psddl_python
#endif // PSDDL_PYTHON_CAMERA_DDL_WRAPPER_H
