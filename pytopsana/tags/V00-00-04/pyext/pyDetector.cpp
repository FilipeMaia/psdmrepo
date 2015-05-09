
#include <boost/python.hpp>
#include "pytopsana/Detector.h"

//-------------------
using namespace pytopsana;

typedef Detector::image_t image_t;
typedef Detector::data_t  data_t;

typedef pytopsana::Detector::pedestals_t     pedestals_t;    // float
typedef pytopsana::Detector::pixel_rms_t     pixel_rms_t;    // float
typedef pytopsana::Detector::pixel_gain_t    pixel_gain_t;   // float
typedef pytopsana::Detector::pixel_mask_t    pixel_mask_t;   // uint16_t
typedef pytopsana::Detector::pixel_bkgd_t    pixel_bkgd_t;   // float
typedef pytopsana::Detector::pixel_status_t  pixel_status_t; // uint16_t
typedef pytopsana::Detector::common_mode_t   common_mode_t;  // double

//typedef pytopsana::Detector::data_i16_t      data_i16_t;     // int16_t

//-------------------

// Create function pointer to each overloaded Detector::calib method
ndarray<const pedestals_t, 1>    (Detector::*peds_1)  (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pedestals;
ndarray<const pixel_rms_t, 1>    (Detector::*prms_1)  (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_rms;
ndarray<const pixel_gain_t, 1>   (Detector::*pgain_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_gain;
ndarray<const pixel_mask_t, 1>   (Detector::*pmask_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_mask;
ndarray<const pixel_bkgd_t, 1>   (Detector::*pbkgd_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_bkgd;
ndarray<const pixel_status_t, 1> (Detector::*pstat_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_status;
ndarray<const common_mode_t, 1>  (Detector::*pcmod_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::common_mode;

//-------------------

ndarray<const int16_t, 1>     (Detector::*pdata_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_1;
ndarray<const int16_t, 2>     (Detector::*pdata_2) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_2;
ndarray<const int16_t, 3>     (Detector::*pdata_3) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_3;
ndarray<const int16_t, 4>     (Detector::*pdata_4) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_4;

ndarray<const uint16_t, 2>    (Detector::*pdata_5) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_uint16_2;
ndarray<const uint16_t, 3>    (Detector::*pdata_6) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_uint16_3;

ndarray<const uint8_t, 2>     (Detector::*pdata_7) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_uint8_2;

//-------------------

ndarray<const double, 1>     (Detector::*pgeo_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_coords_x;
ndarray<const double, 1>     (Detector::*pgeo_2) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_coords_y;
ndarray<const double, 1>     (Detector::*pgeo_3) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_coords_z;

ndarray<const double, 1>     (Detector::*pgeo_4) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_areas;
ndarray<const int, 1>        (Detector::*pgeo_5) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_mask_geo;

ndarray<const unsigned, 1>   (Detector::*pgeo_6) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_indexes_x;
ndarray<const unsigned, 1>   (Detector::*pgeo_7) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_indexes_y;

double                       (Detector::*pgeo_8) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_scale_size;


//-------------------

ndarray<const image_t, 2> (Detector::*img_0) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>, ndarray<const image_t, 1>) = &Detector::get_image;
//ndarray<const float, 2> (Detector::*img_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>, ndarray<const float, 1>&) = &Detector::get_image_float;
//ndarray<const double, 2> (Detector::*img_2) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>, ndarray<const double, 1>&) = &Detector::get_image_double;
//ndarray<const int16_t, 2> (Detector::*img_3) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>, ndarray<const int16_t, 1>&) = &Detector::get_image_int16;
//ndarray<const uint16_t, 2>  (Detector::*img_4) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>, ndarray<const uint16_t, 1>&) = &Detector::get_image_uint16;

//-------------------

void (Detector::*set_1) (const unsigned&) = &Detector::setMode;
void (Detector::*set_2) (const unsigned&) = &Detector::setPrintBits;
void (Detector::*set_3) (const float&)    = &Detector::setDefaultValue;


void (Detector::*print_1) () = &Detector::print;
void (Detector::*print_2) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::print_config;

//-------------------
//ndarray<double,3> (Detector::*calib_2) (ndarray<data_t,3>) = &Detector::calib;
//-------------------

// BOOST wrapper to create pytopsana module that contains the Detector
// python class that calls the C++ Detector methods
// NB: The name of the python module (pytopsana) MUST match the name given in
// PYEXTMOD in the SConscript

BOOST_PYTHON_MODULE(pytopsana_ext)
{    
  using namespace boost::python;

  boost::python::class_<Detector>("Detector", init<const PSEvt::Source, const unsigned&>())
    .def("pedestals",    peds_1)
    .def("pixel_rms",    prms_1)
    .def("pixel_gain",   pgain_1)
    .def("pixel_mask",   pmask_1)
    .def("pixel_bkgd",   pbkgd_1)
    .def("pixel_status", pstat_1)
    .def("common_mode",  pcmod_1)
    .def("data_int16_1", pdata_1)
    .def("data_int16_2", pdata_2)
    .def("data_int16_3", pdata_3)
    .def("data_int16_4", pdata_4)
    .def("data_uint16_2",pdata_5)
    .def("data_uint16_3",pdata_6)
    .def("data_uint8_2", pdata_7)
    .def("pixel_coords_x",  pgeo_1)
    .def("pixel_coords_y",  pgeo_2)
    .def("pixel_coords_z",  pgeo_3)
    .def("pixel_areas",     pgeo_4)
    .def("pixel_mask_geo",  pgeo_5)
    .def("pixel_indexes_x", pgeo_6)
    .def("pixel_indexes_y", pgeo_7)
    .def("pixel_scale_size",pgeo_8)
    .def("get_image",       img_0)
    .def("set_mode",       set_1)
    .def("set_print_bits", set_2)
    .def("set_def_value",  set_3)
    .def("print_members",print_1)
    .def("print_config", print_2)
    .def("inst",         &Detector::str_inst);
}

//-------------------
