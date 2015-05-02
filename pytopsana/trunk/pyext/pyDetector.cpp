
#include <boost/python.hpp>
#include "pytopsana/Detector.h"

using namespace pytopsana;

typedef Detector::data_t data_t;

typedef pytopsana::Detector::pedestals_t     pedestals_t;    // float
typedef pytopsana::Detector::pixel_rms_t     pixel_rms_t;    // float
typedef pytopsana::Detector::pixel_gain_t    pixel_gain_t;   // float
typedef pytopsana::Detector::pixel_mask_t    pixel_mask_t;   // uint16_t
typedef pytopsana::Detector::pixel_bkgd_t    pixel_bkgd_t;   // float
typedef pytopsana::Detector::pixel_status_t  pixel_status_t; // uint16_t
typedef pytopsana::Detector::common_mode_t   common_mode_t;  // double

//typedef pytopsana::Detector::data_i16_t      data_i16_t;     // int16_t


// Create function pointer to each overloaded Detector::calib method
ndarray<const pedestals_t, 1>    (Detector::*peds_1)  (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pedestals;
ndarray<const pixel_rms_t, 1>    (Detector::*prms_1)  (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_rms;
ndarray<const pixel_gain_t, 1>   (Detector::*pgain_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_gain;
ndarray<const pixel_mask_t, 1>   (Detector::*pmask_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_mask;
ndarray<const pixel_bkgd_t, 1>   (Detector::*pbkgd_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_bkgd;
ndarray<const pixel_status_t, 1> (Detector::*pstat_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::pixel_status;
ndarray<const common_mode_t, 1>  (Detector::*pcmod_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::common_mode;

//-------------------
//-------------------
//-------------------
ndarray<const int16_t, 1>     (Detector::*pdata_1) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_1;
ndarray<const int16_t, 2>     (Detector::*pdata_2) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_2;
ndarray<const int16_t, 3>     (Detector::*pdata_3) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_3;
ndarray<const int16_t, 4>     (Detector::*pdata_4) (boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::data_int16_4;
//-------------------
//-------------------
//-------------------

ndarray<double,3> (Detector::*calib_1) (PSEvt::Source, boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>) = &Detector::calib;
ndarray<double,3> (Detector::*calib_2) (ndarray<data_t,3>) = &Detector::calib;

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
    .def("calib",        calib_1)
    .def("calib",        calib_2)
    .def("raw",          &Detector::raw)
    .def("inst",         &Detector::str_inst)
    ;
}
