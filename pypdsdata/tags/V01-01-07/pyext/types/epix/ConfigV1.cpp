//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "AsicConfigV1.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, version)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, runTrigDelay)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, daqTrigDelay)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, dacSetting)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicGR)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicAcq)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicR0)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicPpmat)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicPpbe)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicRoClk)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicGRControl)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicAcqControl)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicR0Control)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicPpmatControl)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicPpbeControl)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicR0ClkControl)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, prepulseR0En)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, adcStreamMode)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, testPatternEnable)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, acqToAsicR0Delay)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicR0ToAsicAcq)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicAcqWidth)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicAcqLToPPmatL)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicRoClkHalfT)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, adcReadsPerPixel)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, adcClkHalfT)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicR0Width)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, adcPipelineDelay)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, prepulseR0Width)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, prepulseR0Delay)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, digitalCardId0)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, digitalCardId1)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, analogCardId0)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, analogCardId1)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, lastRowExclusions)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfAsicsPerRow)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfAsicsPerColumn)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfRowsPerAsic)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfPixelsPerAsicRow)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, baseClockFrequency)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, asicMask)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfRows)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfColumns)
  FUN0_WRAPPER(pypdsdata::Epix::ConfigV1, numberOfAsics)
  PyObject* asics( PyObject* self, PyObject* args );
  PyObject* asicPixelTestArray( PyObject* self, PyObject* args );
  PyObject* asicPixelMaskArray( PyObject* self, PyObject* args );


  PyMethodDef methods[] = {
    { "version",                   version,                   METH_NOARGS, "self.version() -> int\n\nReturns integer number" },
    { "runTrigDelay",              runTrigDelay,              METH_NOARGS, "self.runTrigDelay() -> int\n\nReturns integer number" },
    { "daqTrigDelay",              daqTrigDelay,              METH_NOARGS, "self.daqTrigDelay() -> int\n\nReturns integer number" },
    { "dacSetting",                dacSetting,                METH_NOARGS, "self.dacSetting() -> int\n\nReturns integer number" },
    { "asicGR",                    asicGR,                    METH_NOARGS, "self.asicGR() -> int\n\nReturns integer number" },
    { "asicAcq",                   asicAcq,                   METH_NOARGS, "self.asicAcq() -> int\n\nReturns integer number" },
    { "asicR0",                    asicR0,                    METH_NOARGS, "self.asicR0() -> int\n\nReturns integer number" },
    { "asicPpmat",                 asicPpmat,                 METH_NOARGS, "self.asicPpmat() -> int\n\nReturns integer number" },
    { "asicPpbe",                  asicPpbe,                  METH_NOARGS, "self.asicPpbe() -> int\n\nReturns integer number" },
    { "asicRoClk",                 asicRoClk,                 METH_NOARGS, "self.asicRoClk() -> int\n\nReturns integer number" },
    { "asicGRControl",             asicGRControl,             METH_NOARGS, "self.asicGRControl() -> int\n\nReturns integer number" },
    { "asicAcqControl",            asicAcqControl,            METH_NOARGS, "self.asicAcqControl() -> int\n\nReturns integer number" },
    { "asicR0Control",             asicR0Control,             METH_NOARGS, "self.asicR0Control() -> int\n\nReturns integer number" },
    { "asicPpmatControl",          asicPpmatControl,          METH_NOARGS, "self.asicPpmatControl() -> int\n\nReturns integer number" },
    { "asicPpbeControl",           asicPpbeControl,           METH_NOARGS, "self.asicPpbeControl() -> int\n\nReturns integer number" },
    { "asicR0ClkControl",          asicR0ClkControl,          METH_NOARGS, "self.asicR0ClkControl() -> int\n\nReturns integer number" },
    { "prepulseR0En",              prepulseR0En,              METH_NOARGS, "self.prepulseR0En() -> int\n\nReturns integer number" },
    { "adcStreamMode",             adcStreamMode,             METH_NOARGS, "self.adcStreamMode() -> int\n\nReturns integer number" },
    { "testPatternEnable",         testPatternEnable,         METH_NOARGS, "self.testPatternEnable() -> int\n\nReturns integer number" },
    { "acqToAsicR0Delay",          acqToAsicR0Delay,          METH_NOARGS, "self.acqToAsicR0Delay() -> int\n\nReturns integer number" },
    { "asicR0ToAsicAcq",           asicR0ToAsicAcq,           METH_NOARGS, "self.asicR0ToAsicAcq() -> int\n\nReturns integer number" },
    { "asicAcqWidth",              asicAcqWidth,              METH_NOARGS, "self.asicAcqWidth() -> int\n\nReturns integer number" },
    { "asicAcqLToPPmatL",          asicAcqLToPPmatL,          METH_NOARGS, "self.asicAcqLToPPmatL() -> int\n\nReturns integer number" },
    { "asicRoClkHalfT",            asicRoClkHalfT,            METH_NOARGS, "self.asicRoClkHalfT() -> int\n\nReturns integer number" },
    { "adcReadsPerPixel",          adcReadsPerPixel,          METH_NOARGS, "self.adcReadsPerPixel() -> int\n\nReturns integer number" },
    { "adcClkHalfT",               adcClkHalfT,               METH_NOARGS, "self.adcClkHalfT() -> int\n\nReturns integer number" },
    { "asicR0Width",               asicR0Width,               METH_NOARGS, "self.asicR0Width() -> int\n\nReturns integer number" },
    { "adcPipelineDelay",          adcPipelineDelay,          METH_NOARGS, "self.adcPipelineDelay() -> int\n\nReturns integer number" },
    { "prepulseR0Width",           prepulseR0Width,           METH_NOARGS, "self.prepulseR0Width() -> int\n\nReturns integer number" },
    { "prepulseR0Delay",           prepulseR0Delay,           METH_NOARGS, "self.prepulseR0Delay() -> int\n\nReturns integer number" },
    { "digitalCardId0",            digitalCardId0,            METH_NOARGS, "self.digitalCardId0() -> int\n\nReturns integer number" },
    { "digitalCardId1",            digitalCardId1,            METH_NOARGS, "self.digitalCardId1() -> int\n\nReturns integer number" },
    { "analogCardId0",             analogCardId0,             METH_NOARGS, "self.analogCardId0() -> int\n\nReturns integer number" },
    { "analogCardId1",             analogCardId1,             METH_NOARGS, "self.analogCardId1() -> int\n\nReturns integer number" },
    { "lastRowExclusions",         lastRowExclusions,         METH_NOARGS, "self.lastRowExclusions() -> int\n\nReturns integer number" },
    { "numberOfAsicsPerRow",       numberOfAsicsPerRow,       METH_NOARGS, "self.numberOfAsicsPerRow() -> int\n\nReturns integer number" },
    { "numberOfAsicsPerColumn",    numberOfAsicsPerColumn,    METH_NOARGS, "self.numberOfAsicsPerColumn() -> int\n\nReturns integer number" },
    { "numberOfRowsPerAsic",       numberOfRowsPerAsic,       METH_NOARGS, "self.numberOfRowsPerAsic() -> int\n\nReturns integer number" },
    { "numberOfPixelsPerAsicRow",  numberOfPixelsPerAsicRow,  METH_NOARGS, "self.numberOfPixelsPerAsicRow() -> int\n\nReturns integer number" },
    { "baseClockFrequency",        baseClockFrequency,        METH_NOARGS, "self.baseClockFrequency() -> int\n\nReturns integer number" },
    { "asicMask",                  asicMask,                  METH_NOARGS, "self.asicMask() -> int\n\nReturns integer number" },
    { "numberOfRows",              numberOfRows,              METH_NOARGS, "self.numberOfRows() -> int\n\nReturns integer number" },
    { "numberOfColumns",           numberOfColumns,           METH_NOARGS, "self.numberOfColumns() -> int\n\nReturns integer number" },
    { "numberOfAsics",             numberOfAsics,             METH_NOARGS, "self.numberOfAsics() -> int\n\nReturns integer number" },

    { "asicPixelTestArray",        asicPixelTestArray,        METH_NOARGS, "self.asicPixelTestArray() -> numpy.array\n\nReturns 3-d array" },
    { "asicPixelMaskArray",        asicPixelMaskArray,        METH_NOARGS, "self.asicPixelMaskArray() -> int\n\nReturns integer 3-d array" },
    { "asics",                     asics,                     METH_VARARGS, "self.asics(index: int) -> object\n\nReturns :py:class:`AsicConfigV1` instance" },
    {0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Epix::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Epix::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Epix::ConfigV1::print(std::ostream& str) const
{
  str << "Epix.ConfigV1(version=" << m_obj->version()
      << ", numberOfAsics=" << m_obj->numberOfAsics()
      << ", numberOfRows=" << m_obj->numberOfRows()
      << ", numberOfColumns=" << m_obj->numberOfColumns()
      << ", asicMask=" << int(m_obj->asicMask())
      << ", runTrigDelay=" << int(m_obj->runTrigDelay())
      << ", daqTrigDelay=" << int(m_obj->daqTrigDelay())
      << ", ...)" ;
}

namespace {

PyObject*
asicPixelTestArray( PyObject* self, PyObject* args )
{
  const Pds::Epix::ConfigV1* obj = pypdsdata::Epix::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  ndarray<const uint32_t, 3> data = obj->asicPixelTestArray();

  // NumPy type number
  int typenum = NPY_INT32;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned* shape = data.shape();
  npy_intp dims[3] = { shape[0], shape[1], shape[2] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 3, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
asicPixelMaskArray( PyObject* self, PyObject* args )
{
  const Pds::Epix::ConfigV1* obj = pypdsdata::Epix::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  ndarray<const uint32_t, 3> data = obj->asicPixelMaskArray();

  // NumPy type number
  int typenum = NPY_INT32;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned* shape = data.shape();
  npy_intp dims[3] = { shape[0], shape[1], shape[2] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 3, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
asics( PyObject* self, PyObject* args )
{
  const Pds::Epix::ConfigV1* obj = pypdsdata::Epix::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index;
  if ( not PyArg_ParseTuple( args, "I:Epix.ConfigV1.asics", &index ) ) return 0;

  const Pds::Epix::AsicConfigV1& aconfig = obj->asics(index);
  return pypdsdata::Epix::AsicConfigV1::PyObject_FromPds(const_cast<Pds::Epix::AsicConfigV1*>(&aconfig), self, sizeof aconfig);
}

}
