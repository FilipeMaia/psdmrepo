#ifndef PSANA_CAMERA_DDL_WRAPPER_H
#define PSANA_CAMERA_DDL_WRAPPER_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>
namespace Psana {
namespace Camera {

extern void createWrappers();


/** @class FrameCoord

  Class representing the coordinates of pixels iside the camera frame.
*/


class FrameCoord_Wrapper {
  boost::shared_ptr<FrameCoord> _o;
  FrameCoord* o;
public:
  FrameCoord_Wrapper(boost::shared_ptr<FrameCoord> obj) : _o(obj), o(_o.get()) {}
  FrameCoord_Wrapper(FrameCoord* obj) : o(obj) {}
  uint16_t column() const { return o->column(); }
  uint16_t row() const { return o->row(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
private:
  uint16_t	_column;	/**< Column index (x value). */
  uint16_t	_row;	/**< Row index (y value). */
};

/** @class FrameFccdConfigV1

  This class was never defined/implemented.
*/


class FrameFccdConfigV1_Wrapper {
  boost::shared_ptr<FrameFccdConfigV1> _o;
  FrameFccdConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_FrameFccdConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  FrameFccdConfigV1_Wrapper(boost::shared_ptr<FrameFccdConfigV1> obj) : _o(obj), o(_o.get()) {}
  FrameFccdConfigV1_Wrapper(FrameFccdConfigV1* obj) : o(obj) {}
};

/** @class FrameFexConfigV1

  Class containing configuration data for online frame feature extraction process.
*/


class FrameFexConfigV1_Wrapper {
  boost::shared_ptr<FrameFexConfigV1> _o;
  FrameFexConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_FrameFexConfig /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
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
  FrameFexConfigV1_Wrapper(boost::shared_ptr<FrameFexConfigV1> obj) : _o(obj), o(_o.get()) {}
  FrameFexConfigV1_Wrapper(FrameFexConfigV1* obj) : o(obj) {}
  Camera::FrameFexConfigV1::Forwarding forwarding() const { return o->forwarding(); }
  uint32_t forward_prescale() const { return o->forward_prescale(); }
  Camera::FrameFexConfigV1::Processing processing() const { return o->processing(); }
  const FrameCoord_Wrapper roiBegin() const { return FrameCoord_Wrapper((FrameCoord*) &o->roiBegin()); }
  const FrameCoord_Wrapper roiEnd() const { return FrameCoord_Wrapper((FrameCoord*) &o->roiEnd()); }
  uint32_t threshold() const { return o->threshold(); }
  uint32_t number_of_masked_pixels() const { return o->number_of_masked_pixels(); }
  std::vector<Camera::FrameCoord> masked_pixel_coordinates() const { VEC_CONVERT(o->masked_pixel_coordinates(), Camera::FrameCoord); }
};

/** @class FrameV1

  
*/


class FrameV1_Wrapper {
  boost::shared_ptr<FrameV1> _o;
  FrameV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_Frame /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  FrameV1_Wrapper(boost::shared_ptr<FrameV1> obj) : _o(obj), o(_o.get()) {}
  FrameV1_Wrapper(FrameV1* obj) : o(obj) {}
  uint32_t width() const { return o->width(); }
  uint32_t height() const { return o->height(); }
  uint32_t depth() const { return o->depth(); }
  uint32_t offset() const { return o->offset(); }
  std::vector<uint8_t> _int_pixel_data() const { VEC_CONVERT(o->_int_pixel_data(), uint8_t); }
  PyObject* data8() const { ND_CONVERT(o->data8(), uint8_t, 2); }
  PyObject* data16() const { ND_CONVERT(o->data16(), uint16_t, 2); }
};

/** @class TwoDGaussianV1

  
*/


class TwoDGaussianV1_Wrapper {
  boost::shared_ptr<TwoDGaussianV1> _o;
  TwoDGaussianV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_TwoDGaussian /**< XTC type ID value (from Pds::TypeId class) */ };
  enum { Version = 1 /**< XTC type version number */ };
  TwoDGaussianV1_Wrapper(boost::shared_ptr<TwoDGaussianV1> obj) : _o(obj), o(_o.get()) {}
  TwoDGaussianV1_Wrapper(TwoDGaussianV1* obj) : o(obj) {}
  uint64_t integral() const { return o->integral(); }
  double xmean() const { return o->xmean(); }
  double ymean() const { return o->ymean(); }
  double major_axis_width() const { return o->major_axis_width(); }
  double minor_axis_width() const { return o->minor_axis_width(); }
  double major_axis_tilt() const { return o->major_axis_tilt(); }
};

  class FrameFccdConfigV1_Getter : public Psana::EnvGetter {
  public:
    const std::type_info& getTypeInfo() {
      return typeid(Psana::Camera::FrameFccdConfigV1);
    }
    const char* getTypeName() {
      return "Psana::Camera::FrameFccdConfigV1";
    }
    int getTypeId() {
      return FrameFccdConfigV1::TypeId;
    }
    int getVersion() {
      return FrameFccdConfigV1::Version;
    }
    boost::python::api::object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<FrameFccdConfigV1> result = store.get(src, 0);
      if (! result.get()) {
        return boost::python::api::object();
      }
      return boost::python::api::object(FrameFccdConfigV1_Wrapper(result));
    }
  };

  class FrameFexConfigV1_Getter : public Psana::EnvGetter {
  public:
    const std::type_info& getTypeInfo() {
      return typeid(Psana::Camera::FrameFexConfigV1);
    }
    const char* getTypeName() {
      return "Psana::Camera::FrameFexConfigV1";
    }
    int getTypeId() {
      return FrameFexConfigV1::TypeId;
    }
    int getVersion() {
      return FrameFexConfigV1::Version;
    }
    boost::python::api::object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<FrameFexConfigV1> result = store.get(src, 0);
      if (! result.get()) {
        return boost::python::api::object();
      }
      return boost::python::api::object(FrameFexConfigV1_Wrapper(result));
    }
  };
} // namespace Camera
} // namespace Psana
#endif // PSANA_CAMERA_DDL_WRAPPER_H
