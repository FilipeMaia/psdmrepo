#include <boost/python.hpp>
#include "pytopsana/Detector.h"


using namespace pytopsana;

typedef Detector::data_t data_t;

// Create function pointer to each overloaded Detector::calib method
ndarray<data_t,3> (Detector::*peds_1) (PSEvt::Source, boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>)
            = &Detector::pedestals;

//-------------------
//-------------------
//-------------------

ndarray<double,3> (Detector::*calib_1) (PSEvt::Source, boost::shared_ptr<PSEvt::Event>, boost::shared_ptr<PSEnv::Env>)
            = &Detector::calib;

ndarray<double,3> (Detector::*calib_2) (ndarray<data_t,3>) = &Detector::calib;


// BOOST wrapper to create pytopsana module that contains the Detector
// python class that calls the C++ Detector methods
// NB: The name of the python module (pytopsana) MUST match the name given in
// PYEXTMOD in the SConscript


BOOST_PYTHON_MODULE(pytopsana)
{    
  using namespace boost::python;

  boost::python::class_<Detector>("Detector") // , init<const PSEvt::Source>())
    .def("pedestals", peds_1)
    .def("raw",       &Detector::raw)
    .def("calib",     calib_1)
    .def("calib",     calib_2)
    .def("env",       &Detector::env)
    ;
}
