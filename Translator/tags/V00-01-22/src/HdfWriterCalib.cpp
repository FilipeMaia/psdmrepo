//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	public interface to calib data, as well as internal
//  definition of writers.
//
// Author List:
//      David Schneider
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5/hdf5.h"
#include "boost/make_shared.hpp"
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/cspad.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"
#include "pdscalibdata/CsPadPixelGainV1.h"
#include "pdscalibdata/CsPad2x2PedestalsV1.h"
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"
#include "pdscalibdata/CsPad2x2PixelGainV1.h"

//-----------------------
// External Interface Header --
//-----------------------
#include "Translator/HdfWriterCalib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {
  const char *logger = "HdfWriterCalib";

  void checkConstants() {
    if ((int(pdscalibdata::CsPadPedestalsV1::Quads) != int(pdscalibdata::CsPadPixelStatusV1::Quads)) or \
        (int(pdscalibdata::CsPadPedestalsV1::Sections) != int(pdscalibdata::CsPadPixelStatusV1::Sections)) or \
        (int(pdscalibdata::CsPadPedestalsV1::Columns) != int(pdscalibdata::CsPadPixelStatusV1::Columns)) or \
        (int(pdscalibdata::CsPadPedestalsV1::Rows) != int(pdscalibdata::CsPadPixelStatusV1::Rows)) or \
        (int(pdscalibdata::CsPad2x2PedestalsV1::Sections) != int(pdscalibdata::CsPad2x2PixelStatusV1::Sections)) or \
        (int(pdscalibdata::CsPad2x2PedestalsV1::Columns) != int(pdscalibdata::CsPad2x2PixelStatusV1::Columns)) or \
        (int(pdscalibdata::CsPad2x2PedestalsV1::Rows) != int(pdscalibdata::CsPad2x2PixelStatusV1::Rows)) or \
        (int(pdscalibdata::CsPadPixelGainV1::Quads) != int(pdscalibdata::CsPadPixelStatusV1::Quads)) or \
        (int(pdscalibdata::CsPadPixelGainV1::Sections) != int(pdscalibdata::CsPadPixelStatusV1::Sections)) or \
        (int(pdscalibdata::CsPadPixelGainV1::Columns) != int(pdscalibdata::CsPadPixelStatusV1::Columns)) or \
        (int(pdscalibdata::CsPadPixelGainV1::Rows) != int(pdscalibdata::CsPadPixelStatusV1::Rows)) or \
        (int(pdscalibdata::CsPad2x2PixelGainV1::Sections) != int(pdscalibdata::CsPad2x2PixelStatusV1::Sections)) or \
        (int(pdscalibdata::CsPad2x2PixelGainV1::Columns) != int(pdscalibdata::CsPad2x2PixelStatusV1::Columns)) or \
        (int(pdscalibdata::CsPad2x2PixelGainV1::Rows) != int(pdscalibdata::CsPad2x2PixelStatusV1::Rows))
        )
      {
        throw std::runtime_error("cspad constants differ between pedestal/gain and pixel status pdscalibdata classes");
      }
  } // checkConstants

} // local namespace

namespace Translator {
//   --------------------------------------------------------------
//   -- Internal definition of calibStore types to write to Hdf5 --
//   --------------------------------------------------------------
namespace Calib {

// ------------------------------------------------------------------------
// For each object in the CalibStore that we want to translate, define a
// corresponding struct with functions to construct the
// hdf5 type for the object and a function to construct the data for 
// translation from the pdscalibdata object
// -----------------------------------------------------------------------
struct CsPadCommonModeSubV1 {
  uint32_t mode;
  double data[pdscalibdata::CsPadCommonModeSubV1::DataSize];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPadFilterV1 {
  uint32_t mode;
  double data[pdscalibdata::CsPadFilterV1::DataSize];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPadPixelGainV1 {
  pdscalibdata::CsPadPixelGainV1::pixelGain_t pixelGain[pdscalibdata::CsPadPedestalsV1::Quads]
                                                       [pdscalibdata::CsPadPedestalsV1::Sections]
                                                       [pdscalibdata::CsPadPedestalsV1::Columns]
                                                       [pdscalibdata::CsPadPedestalsV1::Rows];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPad2x2PixelGainV1 {
  pdscalibdata::CsPad2x2PixelGainV1::pixelGain_t pixelGain[pdscalibdata::CsPadPedestalsV1::Sections]
                                                          [pdscalibdata::CsPadPedestalsV1::Columns]
                                                          [pdscalibdata::CsPadPedestalsV1::Rows];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPadPedestalsV1 {
  pdscalibdata::CsPadPedestalsV1::pedestal_t pedestals[pdscalibdata::CsPadPedestalsV1::Quads]
                                                      [pdscalibdata::CsPadPedestalsV1::Sections]
                                                      [pdscalibdata::CsPadPedestalsV1::Columns]
                                                      [pdscalibdata::CsPadPedestalsV1::Rows];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPadPixelStatusV1 {
  pdscalibdata::CsPadPixelStatusV1::status_t status[pdscalibdata::CsPadPixelStatusV1::Quads]
                                                   [pdscalibdata::CsPadPixelStatusV1::Sections]
                                                   [pdscalibdata::CsPadPixelStatusV1::Columns]
                                                   [pdscalibdata::CsPadPixelStatusV1::Rows];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};


struct CsPad2x2PedestalsV1 {
  pdscalibdata::CsPad2x2PedestalsV1::pedestal_t pedestals[pdscalibdata::CsPad2x2PedestalsV1::Columns]
                                                         [pdscalibdata::CsPad2x2PedestalsV1::Rows]
                                                         [pdscalibdata::CsPad2x2PedestalsV1::Sections];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

struct CsPad2x2PixelStatusV1 {
  pdscalibdata::CsPad2x2PixelStatusV1::status_t status[pdscalibdata::CsPad2x2PixelStatusV1::Columns]
                                                      [pdscalibdata::CsPad2x2PixelStatusV1::Rows]
                                                      [pdscalibdata::CsPad2x2PixelStatusV1::Sections];
  static hid_t createHDF5Type(const void *calibObject);
  static const void * fillHdf5WriteBuffer(const void *calibObject);
};

// --------------------------------------------------
// define helper function to construct hdf5 types for 
// members of the pdscalibdata types:
// --------------------------------------------------
enum CalibArrayTypes  { double_1D_CommonDataSize, 
                        double_1D_FilterDataSize, 
                        float_4D_CsPad, 
                        uint16_4D_CsPad,
                        float_3D_CsPad2x2, 
                        uint16_3D_CsPad2x2 };

hid_t calibStoreCommonTypes(CalibArrayTypes whichArrayType) {
  static bool firstCall = true;
  static hid_t hid_double_1D_CommonDataSize = -1;
  static hid_t hid_double_1D_FilterDataSize = -1;
  static hid_t hid_float_4D_CsPad = -1;
  static hid_t hid_uint16_4D_CsPad = -1;
  static hid_t hid_float_3D_CsPad2x2 = -1;
  static hid_t hid_uint16_3D_CsPad2x2 = -1;
  
  const int DIM = 4;
  static const hsize_t cspadDims[DIM] = {pdscalibdata::CsPadPedestalsV1::Quads,
                                         pdscalibdata::CsPadPedestalsV1::Sections,
                                         pdscalibdata::CsPadPedestalsV1::Columns,
                                         pdscalibdata::CsPadPedestalsV1::Rows};
  
  const int DIM2x2 = 3;
  static const hsize_t cspad2x2Dims[DIM2x2] = {pdscalibdata::CsPad2x2PedestalsV1::Columns,
                                               pdscalibdata::CsPad2x2PedestalsV1::Rows,
                                               pdscalibdata::CsPad2x2PedestalsV1::Sections};
  
  static const hsize_t dimCommonModeSubV1[1] = {pdscalibdata::CsPadCommonModeSubV1::DataSize};
  static const hsize_t dimFilterV1[1] = {pdscalibdata::CsPadFilterV1::DataSize};
  
  if (firstCall) {
    firstCall = false;
    checkConstants();
    hid_double_1D_CommonDataSize = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, dimCommonModeSubV1); 
    hid_double_1D_FilterDataSize = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, dimFilterV1);
    hid_float_4D_CsPad = H5Tarray_create2(H5T_NATIVE_FLOAT, DIM, cspadDims);
    hid_uint16_4D_CsPad = H5Tarray_create2(H5T_NATIVE_UINT16, DIM, cspadDims);
    hid_float_3D_CsPad2x2 = H5Tarray_create2(H5T_NATIVE_FLOAT, DIM2x2, cspad2x2Dims);
    hid_uint16_3D_CsPad2x2 = H5Tarray_create2(H5T_NATIVE_UINT16, DIM2x2, cspad2x2Dims);
    
    if (hid_double_1D_CommonDataSize < 0 or hid_double_1D_FilterDataSize < 0) {
      MsgLog(logger, error, "Failed to make h5 types for calib common mode or"
             << "filter data parameters." 
             << " hid_double_1D_CommonDataSize = " << hid_double_1D_CommonDataSize 
             << " hid_double_1D_FilterDataSize = " << hid_double_1D_FilterDataSize);
    }
    if (hid_float_4D_CsPad < 0 or hid_uint16_4D_CsPad < 0 or            \
        hid_float_3D_CsPad2x2 < 0 or hid_uint16_3D_CsPad2x2 < 0) {
      MsgLog(logger,error, "Failed to make h5 types for calib2x2 pedstals or pixel2x2 status"
             << " hid_float_3D_CsPad2x2 = " << hid_float_3D_CsPad2x2
             << " hid_uint16_3D_CsPad2x2 = " << hid_uint16_3D_CsPad2x2);
    }
  }
  
  switch (whichArrayType) {
  case double_1D_CommonDataSize:
    return hid_double_1D_CommonDataSize;
  case double_1D_FilterDataSize:
    return hid_double_1D_FilterDataSize;
  case float_4D_CsPad:
    return hid_float_4D_CsPad;
  case uint16_4D_CsPad:
    return hid_uint16_4D_CsPad;
  case float_3D_CsPad2x2:
    return hid_float_3D_CsPad2x2;
  case uint16_3D_CsPad2x2:
    return hid_uint16_3D_CsPad2x2;
  }
  // error if we reach this far
  return -1;
}

// ---------------------------------------------
// Functions that create and hdf5 types and fill buffers for translation
// of the pdscalibdata types

hid_t CsPadCommonModeSubV1::createHDF5Type(const void *calibObject) {
  bool firstCall = true;
  static hid_t h5type = -1;
  if (firstCall) {
    firstCall = false;
    h5type = H5Tcreate(H5T_COMPOUND, sizeof(Translator::Calib::CsPadCommonModeSubV1));
    herr_t status1 = H5Tinsert(h5type, "mode", 
                               offsetof(Translator::Calib::CsPadCommonModeSubV1,mode), 
                               H5T_NATIVE_UINT32);
    herr_t status2 = H5Tinsert(h5type, "data", 
                               offsetof(Translator::Calib::CsPadCommonModeSubV1,data), 
                               calibStoreCommonTypes(double_1D_CommonDataSize));
    if ((h5type < 0) or (status1 < 0) or (status2<0)) {
      MsgLog(logger,error,"unable to create " 
             << "Translator::Calib::CsPadCommonModeSubV1 compound type."
             << " h5type = " << h5type
             << " status1= " << status1
             << " status2= " << status2);
    } else {
      MsgLog(logger,trace,"Created hdf5 type for Translator::Calib::CsPadCommonModeSubV1, h5type=" << h5type);  
    }
  }
  return h5type;
}

const void * CsPadCommonModeSubV1::fillHdf5WriteBuffer(const void *calibObject) {
  static Translator::Calib::CsPadCommonModeSubV1 buffer;
  const pdscalibdata::CsPadCommonModeSubV1 * obj = (const pdscalibdata::CsPadCommonModeSubV1 *)calibObject; 
  buffer.mode = obj->mode();
  const double *p = obj->data();
  for (int i = 0; i < pdscalibdata::CsPadCommonModeSubV1::DataSize; ++i) {
    buffer.data[i]=p[i];
  }
  return &buffer;
}

hid_t CsPadFilterV1::createHDF5Type(const void *calibObject) {
  bool firstCall = true;
  static hid_t h5type = -1;
  if (firstCall) {
    firstCall = false;
    h5type = H5Tcreate(H5T_COMPOUND, sizeof(Translator::Calib::CsPadFilterV1));
    herr_t status1 = H5Tinsert(h5type, "mode", 
                               offsetof(Translator::Calib::CsPadFilterV1,mode), 
                               H5T_NATIVE_UINT32);
    herr_t status2 = H5Tinsert(h5type, "data", 
                               offsetof(Translator::Calib::CsPadFilterV1,data), 
                               calibStoreCommonTypes(double_1D_FilterDataSize));
    if ((h5type < 0) or (status1 < 0) or (status2<0)) {
      MsgLog(logger,error,"unable to create " 
             << "Translator::Calib::CsPadFilterV1 compound type."
             << " h5type = " << h5type
             << " status1= " << status1
             << " status2= " << status2);
    } else {
      MsgLog(logger,trace,"Created hdf5 type for Translator::Calib::CsPadFilterV1, h5type=" << h5type);  
    }
  }
  return h5type;
}

const void * CsPadFilterV1::fillHdf5WriteBuffer(const void *calibObject) {
  static Translator::Calib::CsPadFilterV1 buffer;
  const pdscalibdata::CsPadFilterV1 * obj = (const pdscalibdata::CsPadFilterV1 *)calibObject; 
  buffer.mode = obj->mode();
  const double *p = obj->data();
  for (int i = 0; i < pdscalibdata::CsPadFilterV1::DataSize; ++i) {
    buffer.data[i]=p[i];
  }
  return &buffer;
}

hid_t CsPadPedestalsV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(float_4D_CsPad);
}

const void * CsPadPedestalsV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPadPedestalsV1 * obj = (const pdscalibdata::CsPadPedestalsV1 *)calibObject; 
  const float * p = obj->pedestals().data();
  return p;
}

hid_t CsPadPixelStatusV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(uint16_4D_CsPad);
}

const void * CsPadPixelStatusV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPadPixelStatusV1 * obj = (const pdscalibdata::CsPadPixelStatusV1 *)calibObject; 
  const uint16_t * p = obj->status().data();
  return p;
}

hid_t CsPad2x2PedestalsV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(float_3D_CsPad2x2);
}

const void * CsPad2x2PedestalsV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPad2x2PedestalsV1 * obj = (const pdscalibdata::CsPad2x2PedestalsV1 *)calibObject; 
  const float * p = obj->pedestals().data();
  return p;
}

hid_t CsPad2x2PixelStatusV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(uint16_3D_CsPad2x2);
}

const void * CsPad2x2PixelStatusV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPad2x2PixelStatusV1 * obj = (const pdscalibdata::CsPad2x2PixelStatusV1 *)calibObject; 
  const uint16_t * p = obj->status().data();
  return p;
}

hid_t CsPad2x2PixelGainV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(float_3D_CsPad2x2);
}

const void * CsPad2x2PixelGainV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPad2x2PixelGainV1 * obj = (const pdscalibdata::CsPad2x2PixelGainV1 *)calibObject; 
  const float * p = obj->pixelGains().data();
  return p;
}

hid_t CsPadPixelGainV1::createHDF5Type(const void *calibObject) {
  return calibStoreCommonTypes(float_4D_CsPad);
}

const void * CsPadPixelGainV1::fillHdf5WriteBuffer(const void *calibObject) {
  const pdscalibdata::CsPadPixelGainV1 * obj = (const pdscalibdata::CsPadPixelGainV1 *)calibObject; 
  const float * p = obj->pixelGains().data();
  return p;
}

} // Calib namespace

// --------------------------------------------
// public interface of this code
// ---------------------------------------
void getHdfWritersForCalibStore(std::vector< boost::shared_ptr<HdfWriterNew> > & calibStoreWriters) {
  calibStoreWriters.clear();
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPadCommonModeSubV1),
                                                     "data",
                                                     Calib::CsPadCommonModeSubV1::createHDF5Type,
                                                     Calib::CsPadCommonModeSubV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPadFilterV1),
                                                     "data",
                                                     Calib::CsPadFilterV1::createHDF5Type,
                                                     Calib::CsPadFilterV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPadPedestalsV1),
                                                     "pedestals",
                                                     Calib::CsPadPedestalsV1::createHDF5Type,
                                                     Calib::CsPadPedestalsV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPadPixelStatusV1),
                                                     "status",
                                                     Calib::CsPadPixelStatusV1::createHDF5Type,
                                                     Calib::CsPadPixelStatusV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPad2x2PedestalsV1),
                                                     "pedestals",
                                                     Calib::CsPad2x2PedestalsV1::createHDF5Type,
                                                     Calib::CsPad2x2PedestalsV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPad2x2PixelStatusV1),
                                                     "status",
                                                     Calib::CsPad2x2PixelStatusV1::createHDF5Type,
                                                     Calib::CsPad2x2PixelStatusV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPadPixelGainV1),
                                                     "pixel_gain",
                                                     Calib::CsPadPixelGainV1::createHDF5Type,
                                                     Calib::CsPadPixelGainV1::fillHdf5WriteBuffer));
  calibStoreWriters.push_back(
                    boost::make_shared<HdfWriterNew>(&typeid(pdscalibdata::CsPad2x2PixelGainV1),
                                                     "pixel_gain",
                                                     Calib::CsPad2x2PixelGainV1::createHDF5Type,
                                                     Calib::CsPad2x2PixelGainV1::fillHdf5WriteBuffer));
}

void getType2CalibTypesMap(Type2CalibTypesMap & type2calibTypeMap) {
  type2calibTypeMap.clear();
  std::vector<const std::type_info *> cspadCalib, cspad2x2Calib;
  cspad2x2Calib.push_back(&typeid(pdscalibdata::CsPad2x2PedestalsV1));
  cspad2x2Calib.push_back(&typeid(pdscalibdata::CsPad2x2PixelStatusV1));
  cspad2x2Calib.push_back(&typeid(pdscalibdata::CsPad2x2PixelGainV1));
  cspad2x2Calib.push_back(&typeid(pdscalibdata::CsPadCommonModeSubV1));
  
  cspadCalib.push_back(&typeid(pdscalibdata::CsPadPedestalsV1));
  cspadCalib.push_back(&typeid(pdscalibdata::CsPadPixelStatusV1));
  cspadCalib.push_back(&typeid(pdscalibdata::CsPadPixelGainV1));
  cspadCalib.push_back(&typeid(pdscalibdata::CsPadCommonModeSubV1));
  cspadCalib.push_back(&typeid(pdscalibdata::CsPadFilterV1));
  
  type2calibTypeMap[&typeid(Psana::CsPad::DataV2)] = cspadCalib;
  type2calibTypeMap[&typeid(Psana::CsPad::DataV1)] = cspadCalib;
  
  type2calibTypeMap[&typeid(Psana::CsPad2x2::ElementV1)] = cspad2x2Calib;
} 


} // Translator namespace

