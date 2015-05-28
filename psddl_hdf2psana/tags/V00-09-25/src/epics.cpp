//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/epics.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/Utils.h"
#include "MsgLogger/MsgLogger.h"
#include "psddl_hdf2psana/epics.ddl.h"
#include "psddl_psana/EpicsLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  using namespace psddl_hdf2psana::Epics;

  const char logger[] = "psddl_hdf2psana.Epics";

  boost::shared_ptr<Psana::Epics::EpicsPvHeader>
  readEpics(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr);

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

// bleh
#define ATTR_SPEC(TYPE, MEMBER, OFFSET) #MEMBER, OFFSET + offsetof(TYPE, MEMBER), hdf5pp::TypeTraits<typeof(((TYPE*)0)->MEMBER)>::native_type()

using namespace Psana::EpicsLib;

namespace psddl_hdf2psana {
namespace Epics {

// few methods that generator does not know how to implement
// (and they are not really necessary)
uint16_t 
EpicsPvHeader_v0::status() const
{
  return 0;
}
uint16_t 
EpicsPvHeader_v0::severity() const 
{
  return 0;
}




// special deleter for boost shared pointers
template <typename T>
void bufdelete(const T* buf) {
  delete [] (const char*)buf;
}

// special type meaning enum
struct EpicsEnumTag {};

template <typename Type> 
struct Limits {
  Type    upper_disp_limit;
  Type    lower_disp_limit;
  Type    upper_alarm_limit;
  Type    upper_warning_limit;
  Type    lower_warning_limit;
  Type    lower_alarm_limit;
  Type    upper_ctrl_limit;
  Type    lower_ctrl_limit;
};

template <typename T>
struct dbr_time {
};

template <>
struct dbr_time<const char*> {
  typedef Psana::Epics::dbr_time_string psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<EpicsEnumTag> {
  typedef Psana::Epics::dbr_time_enum psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<uint8_t> {
  typedef Psana::Epics::dbr_time_char psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<int16_t> {
  typedef Psana::Epics::dbr_time_short psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<int32_t> {
  typedef Psana::Epics::dbr_time_long psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<float> {
  typedef Psana::Epics::dbr_time_float psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};

template <>
struct dbr_time<double> {
  typedef Psana::Epics::dbr_time_double psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  ns_epicsTimeStamp_v0::dataset_data stamp;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, stamp); }
};




template <typename T>
struct dbr_ctrl {
};

template <>
struct dbr_ctrl<const char*> {
  typedef Psana::Epics::dbr_sts_string psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity); }
};

template <>
struct dbr_ctrl<EpicsEnumTag> {
  typedef Psana::Epics::dbr_ctrl_enum psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  int16_t     no_str;
  char        strs[MAX_ENUM_STATES][MAX_ENUM_STRING_SIZE];
  
  operator psana_dbr_t() const { return psana_dbr_t(status, severity, no_str, &strs[0][0]); }
};

template <>
struct dbr_ctrl<uint8_t> {
  typedef Psana::Epics::dbr_ctrl_char psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  char        units[MAX_UNITS_SIZE];
  Limits<uint8_t>   limits;
  
  operator psana_dbr_t() const { 
    return psana_dbr_t(status, severity, units, limits.upper_disp_limit, limits.lower_disp_limit, 
        limits.upper_alarm_limit, limits.upper_warning_limit, limits.lower_warning_limit, limits.lower_alarm_limit, 
        limits.upper_ctrl_limit, limits.lower_ctrl_limit); 
  }
};

template <>
struct dbr_ctrl<int16_t> {
  typedef Psana::Epics::dbr_ctrl_short psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  char        units[MAX_UNITS_SIZE];
  Limits<int16_t>   limits;

  operator psana_dbr_t() const { 
    return psana_dbr_t(status, severity, units, limits.upper_disp_limit, limits.lower_disp_limit, 
        limits.upper_alarm_limit, limits.upper_warning_limit, limits.lower_warning_limit, limits.lower_alarm_limit, 
        limits.upper_ctrl_limit, limits.lower_ctrl_limit); 
  }
};

template <>
struct dbr_ctrl<int32_t> {
  typedef Psana::Epics::dbr_ctrl_long psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  char        units[MAX_UNITS_SIZE];
  Limits<int32_t>   limits;

  operator psana_dbr_t() const { 
    return psana_dbr_t(status, severity, units, limits.upper_disp_limit, limits.lower_disp_limit, 
        limits.upper_alarm_limit, limits.upper_warning_limit, limits.lower_warning_limit, limits.lower_alarm_limit, 
        limits.upper_ctrl_limit, limits.lower_ctrl_limit); 
  }
};

template <>
struct dbr_ctrl<float> {
  typedef Psana::Epics::dbr_ctrl_float psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  int16_t     precision;
  char        units[MAX_UNITS_SIZE];
  Limits<float> limits;

  operator psana_dbr_t() const { 
    return psana_dbr_t(status, severity, precision, units, limits.upper_disp_limit, limits.lower_disp_limit, 
        limits.upper_alarm_limit, limits.upper_warning_limit, limits.lower_warning_limit, limits.lower_alarm_limit, 
        limits.upper_ctrl_limit, limits.lower_ctrl_limit); 
  }
};

template <>
struct dbr_ctrl<double> {
  typedef Psana::Epics::dbr_ctrl_double psana_dbr_t;

  int16_t     status;
  int16_t     severity;
  int16_t     precision;
  char        units[MAX_UNITS_SIZE];
  Limits<double> limits;

  operator psana_dbr_t() const { 
    return psana_dbr_t(status, severity, precision, units, limits.upper_disp_limit, limits.lower_disp_limit, 
        limits.upper_alarm_limit, limits.upper_warning_limit, limits.lower_warning_limit, limits.lower_alarm_limit, 
        limits.upper_ctrl_limit, limits.lower_ctrl_limit); 
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename Type>
struct CompoundTypeDefs {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert("value", offset, hdf5pp::TypeTraits<Type>::native_type(extra));
  }

};

template <>
struct CompoundTypeDefs<const char*> {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    hdf5pp::Type oneType = hdf5pp::TypeTraits<const char*>::native_type(MAX_STRING_SIZE);
    if (extra == 0) {
      type.insert("value", offset, oneType);
    } else {
      type.insert("value", offset, hdf5pp::ArrayType::arrayType(oneType, extra));
    }
  }

};

template <typename Type>
struct CompoundTypeDefs<Limits<Type>  > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset) {
    type.insert(ATTR_SPEC(Limits<Type>, upper_disp_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, lower_disp_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, upper_alarm_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, upper_warning_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, lower_warning_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, lower_alarm_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, upper_ctrl_limit, offset));
    type.insert(ATTR_SPEC(Limits<Type>, lower_ctrl_limit, offset));
  }

};

template <typename T>
struct CompoundTypeDefs<dbr_time<T> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_time<T>, status, offset));
    type.insert(ATTR_SPEC(dbr_time<T>, severity, offset));
    type.insert(ATTR_SPEC(dbr_time<T>, stamp, offset));
  }

};

template <typename T>
struct CompoundTypeDefs<dbr_ctrl<T> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_ctrl<T>, status, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<T>, severity, offset));
    type.insert("units", offset+offsetof(dbr_ctrl<T>, units), hdf5pp::TypeTraits<const char*>::native_type(MAX_UNITS_SIZE));
    CompoundTypeDefs<Limits<T> >::defineFields(type, offset+offsetof(dbr_ctrl<T>, limits));
  }

};

template <>
struct CompoundTypeDefs<dbr_ctrl<const char*> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_ctrl<const char*>, status, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<const char*>, severity, offset));
  }

};

template <>
struct CompoundTypeDefs<dbr_ctrl<EpicsEnumTag> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_ctrl<EpicsEnumTag>, status, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<EpicsEnumTag>, severity, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<EpicsEnumTag>, no_str, offset));
    type.insert("strs", offset+offsetof(dbr_ctrl<EpicsEnumTag>, strs), hdf5pp::TypeTraits<const char*>::native_type(MAX_ENUM_STRING_SIZE), extra);
  }

};

template <>
struct CompoundTypeDefs<dbr_ctrl<float> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_ctrl<float>, status, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<float>, severity, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<float>, precision, offset));
    type.insert("units", offset+offsetof(dbr_ctrl<float>, units), hdf5pp::TypeTraits<const char*>::native_type(MAX_UNITS_SIZE));
    CompoundTypeDefs<Limits<float> >::defineFields(type, offset+offsetof(dbr_ctrl<float>, limits));
  }

};

template <>
struct CompoundTypeDefs<dbr_ctrl<double> > {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset, int extra) {
    type.insert(ATTR_SPEC(dbr_ctrl<double>, status, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<double>, severity, offset));
    type.insert(ATTR_SPEC(dbr_ctrl<double>, precision, offset));
    type.insert("units", offset+offsetof(dbr_ctrl<double>, units), hdf5pp::TypeTraits<const char*>::native_type(MAX_UNITS_SIZE));
    CompoundTypeDefs<Limits<double> >::defineFields(type, offset+offsetof(dbr_ctrl<double>, limits));
  }

};

template <>
struct CompoundTypeDefs<ns_EpicsPvHeader_v0::dataset_data> {

  static void defineFields(hdf5pp::CompoundType& type, unsigned offset) {
    type.insert(ATTR_SPEC(ns_EpicsPvHeader_v0::dataset_data, pvId, offset));
    type.insert(ATTR_SPEC(ns_EpicsPvHeader_v0::dataset_data, dbrType, offset));
    type.insert(ATTR_SPEC(ns_EpicsPvHeader_v0::dataset_data, numElements, offset));
  }

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename dbr_type>
struct ValueType {
  typedef typename DBRTypeTraits<dbr_type>::value_type type; 
};

template <>
struct ValueType<Psana::Epics::dbr_time_string> {
  typedef const char* type; 
};

template <>
struct ValueType<Psana::Epics::dbr_sts_string> {
  typedef const char* type; 
};

// DBR can be any dbr_time<T>
template <typename DBR>
struct dataset_epics_time {
  
  typedef typename DBR::psana_dbr_t dbr_type;
  typedef typename DBRTypeTraits<dbr_type>::value_type value_type;
  typedef typename ValueType<dbr_type>::type hdf5_value_type;
  
  // create instance capable of holding specified number of elements
  static boost::shared_ptr<dataset_epics_time> make_dataset_epics(unsigned nElem) {
    if (nElem < 1) nElem = 1;
    size_t size = sizeof(dataset_epics_time) + sizeof(value_type)*(nElem-1);
    char* buf = new char[size];
    dataset_epics_time* ds = new (buf) dataset_epics_time;
    return boost::shared_ptr<dataset_epics_time>(ds, bufdelete<dataset_epics_time>);
  }

  // Get the HDF5 type
  static hdf5pp::Type native_type(unsigned nElem, int extra = 0) {
    
    // need to know size of the full structure
    if (nElem < 1) nElem = 1;
    size_t size = sizeof(dataset_epics_time) + sizeof(value_type)*(nElem-1);

    hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(size);
    CompoundTypeDefs<ns_EpicsPvHeader_v0::dataset_data>::defineFields(type, offsetof(dataset_epics_time, header));
    CompoundTypeDefs<DBR>::defineFields(type, offsetof(dataset_epics_time, dbr), extra);
    CompoundTypeDefs<hdf5_value_type>::defineFields(type, offsetof(dataset_epics_time, value), nElem > 1 ? nElem : 0);
    
    return type;
  }

  ns_EpicsPvHeader_v0::dataset_data header;
  DBR                               dbr;
  value_type                        value[1];
};

// DBR can be any dbr_ctrl<T>
template <typename DBR>
struct dataset_epics_ctrl {
  
  typedef typename DBR::psana_dbr_t dbr_type;
  typedef typename DBRTypeTraits<dbr_type>::value_type value_type;
  typedef typename ValueType<dbr_type>::type hdf5_value_type;

  // create instance capable of holding specified number of elements
  static boost::shared_ptr<dataset_epics_ctrl> make_dataset_epics(unsigned nElem) {
    if (nElem < 1) nElem = 1;
    size_t size = sizeof(dataset_epics_ctrl) + sizeof(value_type)*(nElem-1);
    char* buf = new char[size];
    dataset_epics_ctrl* ds = new (buf) dataset_epics_ctrl;
    return boost::shared_ptr<dataset_epics_ctrl>(ds, bufdelete<dataset_epics_ctrl>);
  }

  // Get the HDF5 type
  static hdf5pp::Type native_type(unsigned nElem, int extra = 0) {
    
    // need to know size of the full structure
    if (nElem < 1) nElem = 1;
    size_t size = sizeof(dataset_epics_ctrl) + sizeof(value_type)*(nElem-1);

    hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(size);
    CompoundTypeDefs<ns_EpicsPvHeader_v0::dataset_data>::defineFields(type, offsetof(dataset_epics_ctrl, header));
    type.insert("pvname", offsetof(dataset_epics_ctrl, pvname), hdf5pp::TypeTraits<const char*>::native_type(iMaxPvNameLength));
    CompoundTypeDefs<DBR>::defineFields(type, offsetof(dataset_epics_ctrl, dbr), extra);
    CompoundTypeDefs<hdf5_value_type>::defineFields(type, offsetof(dataset_epics_ctrl, value), nElem > 1 ? nElem : 0);
    
    return type;
  }

  ns_EpicsPvHeader_v0::dataset_data header;
  char                              pvname[iMaxPvNameLength];
  DBR                               dbr;
  value_type                        value[1];
};


// =============================================================


// DBR can be any dbr_time<T> or dbr_ctrl<T>, ds_type is one of the dataset_epics_ctrl<DBR> or dataset_epics_time<DBR>
template <typename DBR, typename ds_type>
class EpicsPvHdr : public DBRTypeTraits<typename DBR::psana_dbr_t>::pv_type {
public:
  
  typedef typename DBR::psana_dbr_t dbr_type;
  
  EpicsPvHdr(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : m_ds(ds), m_idx(idx), m_hdr(hdr) {}

  /** PV ID number assigned by DAQ. */
  virtual int16_t pvId() const { return m_hdr.pvId; }
  
  /** DBR structure type. */
  virtual int16_t dbrType() const { return m_hdr.dbrType; }
  
  /** Number of elements in EPICS DBR structure */
  virtual int16_t numElements() const { return m_hdr.numElements; }
  
  /** Returns 1 if PV is one of CTRL types, 0 otherwise. */
  virtual uint8_t isCtrl() const {
    return m_hdr.dbrType >= DBR_CTRL_STRING and m_hdr.dbrType <= DBR_CTRL_DOUBLE;
  }
  
  /** Returns 1 if PV is one of TIME types, 0 otherwise. */
  virtual uint8_t isTime() const {
    return m_hdr.dbrType >= DBR_TIME_STRING and m_hdr.dbrType <= DBR_TIME_DOUBLE;
  }
  
  /** Returns status value for the PV. */
  virtual uint16_t status() const {
    if (not m_ds_data) read_ds_data();
    return m_dbr.status();
  }
  
  /** Returns severity value for the PV. */
  virtual uint16_t severity() const {
    if (not m_ds_data) read_ds_data();
    return m_dbr.severity();
  }
  
  virtual const dbr_type& dbr() const {
    if (not m_ds_data) read_ds_data();
    return m_dbr;
  }

protected:
      
  mutable hdf5pp::DataSet m_ds;
  uint64_t m_idx;
  ns_EpicsPvHeader_v0::dataset_data m_hdr;

  mutable boost::shared_ptr<ds_type> m_ds_data;
  mutable dbr_type m_dbr;
  
  virtual void read_ds_data() const {
    m_ds_data = ds_type::make_dataset_epics(m_hdr.numElements);
    
    hdf5pp::DataSpace file_dsp = m_ds.dataSpace();
    file_dsp.select_single(m_idx);
    m_ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, m_ds_data.get(), ds_type::native_type(m_hdr.numElements));
    m_dbr = dbr_type(m_ds_data->dbr);
  }
};


// DBR can be any dbr_ctrl<T>
template <typename DBR>
class EpicsPvCtrlHdr : public EpicsPvHdr<DBR, dataset_epics_ctrl<DBR> > {
public:

  EpicsPvCtrlHdr(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvHdr<DBR, dataset_epics_ctrl<DBR> >(ds, idx, hdr) {}
  
  /** PV name. */
  virtual const char* pvName() const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->pvname;
  }
 
};

// DBR can be any dbr_time<T>
template <typename DBR>
class EpicsPvTimeHdr : public EpicsPvHdr<DBR, dataset_epics_time<DBR> > {
public:

  EpicsPvTimeHdr(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvHdr<DBR, dataset_epics_time<DBR> >(ds, idx, hdr) {}
  
  /** PV name. */
  virtual Psana::Epics::epicsTimeStamp stamp() const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_dbr.stamp();
  }
 
};


// DBR can be any dbr_ctrl<T>
template <typename DBR>
class EpicsPvCtrl : public EpicsPvCtrlHdr<DBR> {
public:

  typedef typename DBR::psana_dbr_t dbr_type;
  typedef typename DBRTypeTraits<dbr_type>::value_type value_type;
  
  EpicsPvCtrl(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvCtrlHdr<DBR>(ds, idx, hdr) {}
  
  virtual ndarray<const value_type, 1> data() const {
    if (not this->m_ds_data) this->read_ds_data();
    return make_ndarray<value_type>(boost::shared_ptr<value_type>(this->m_ds_data, this->m_ds_data->value), this->m_hdr.numElements);
  }
  
  virtual value_type value(uint32_t i) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i];
  }
};

// specialization for strings
template <>
class EpicsPvCtrl<dbr_ctrl<const char*> > : public EpicsPvCtrlHdr<dbr_ctrl<const char*> > {
public:

  EpicsPvCtrl(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvCtrlHdr<dbr_ctrl<const char*> >(ds, idx, hdr) {}
  
  virtual const char* data(uint32_t i0) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i0];
  }
  virtual const char* value(uint32_t i) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i];
  }
  
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  virtual std::vector<int> data_shape() const {
    return std::vector<int>(1, this->m_hdr.numElements);
  }
};


// specialization for enum type
template <>
class EpicsPvCtrl<dbr_ctrl<EpicsEnumTag> > : public EpicsPvCtrlHdr<dbr_ctrl<EpicsEnumTag> > {
public:

  typedef dbr_ctrl<EpicsEnumTag>::psana_dbr_t dbr_type;
  typedef DBRTypeTraits<dbr_type>::value_type value_type;
  typedef dataset_epics_ctrl<dbr_ctrl<EpicsEnumTag> > ds_type;

  EpicsPvCtrl(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr)
    : EpicsPvCtrlHdr<dbr_ctrl<EpicsEnumTag> >(ds, idx, hdr) {}

  virtual ndarray<const value_type, 1> data() const {
    if (not this->m_ds_data) this->read_ds_data();
    return make_ndarray<value_type>(boost::shared_ptr<value_type>(this->m_ds_data, this->m_ds_data->value), this->m_hdr.numElements);
  }

  virtual value_type value(uint32_t i) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i];
  }

private:

  virtual void read_ds_data() const {
    m_ds_data = ds_type::make_dataset_epics(m_hdr.numElements);

    hdf5pp::DataSpace file_dsp = m_ds.dataSpace();
    file_dsp.select_single(m_idx);

    // need to read no_str first
    int16_t no_str;
    hdf5pp::CompoundType n_type = hdf5pp::CompoundType::compoundType(sizeof(int16_t));
    n_type.insert("no_str", 0, hdf5pp::TypeTraits<int16_t>::native_type());
    m_ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, &no_str, n_type);

    m_ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, m_ds_data.get(), ds_type::native_type(m_hdr.numElements, no_str));

    m_dbr = dbr_type(m_ds_data->dbr);
  }
};


// DBR can be any dbr_time<T>
template <typename DBR>
class EpicsPvTime : public EpicsPvTimeHdr<DBR> {
public:

  typedef typename DBR::psana_dbr_t dbr_type;
  typedef typename DBRTypeTraits<dbr_type>::value_type value_type;

  EpicsPvTime(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvTimeHdr<DBR>(ds, idx, hdr) {}
  
  virtual ndarray<const value_type, 1> data() const {
    if (not this->m_ds_data) this->read_ds_data();
    return make_ndarray<value_type>(boost::shared_ptr<value_type>(this->m_ds_data, this->m_ds_data->value), this->m_hdr.numElements);
  }
  
  virtual value_type value(uint32_t i) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i];
  }
};

// specialization for strings
template <>
class EpicsPvTime<dbr_time<const char*> > : public EpicsPvTimeHdr<dbr_time<const char*> > {
public:

  EpicsPvTime(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr) 
    : EpicsPvTimeHdr<dbr_time<const char*> >(ds, idx, hdr) {}
  
  virtual const char* data(uint32_t i0) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i0];
  }
  virtual const char* value(uint32_t i) const {
    if (not this->m_ds_data) this->read_ds_data();
    return this->m_ds_data->value[i];
  }
  
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  virtual std::vector<int> data_shape() const {
    return std::vector<int>(1, this->m_hdr.numElements);
  }
};


/*
 *  Read data from specified group and convert them into EPICS object
 */
boost::shared_ptr<Psana::Epics::EpicsPvHeader>
readEpics(const hdf5pp::DataSet& ds, int64_t idx)
{
  MsgLog(logger, debug, "readEpics: dataset = " << ds.name() << " idx = " << idx);
  
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> result;

  // first we need to read a header to guess the type/size of stored data
  const boost::shared_ptr<ns_EpicsPvHeader_v0::dataset_data>& hdr =
      hdf5pp::Utils::readDataSet<ns_EpicsPvHeader_v0::dataset_data>(ds, idx);

  return ::readEpics(ds, idx, *hdr);
}

/*
 *  Read data from specified group and convert them into EPICS object
 */
boost::shared_ptr<Psana::Epics::EpicsPvHeader>
readEpics(const hdf5pp::DataSet& ds, int64_t idx, const Psana::Epics::EpicsPvHeader& pvhdr)
{
  MsgLog(logger, debug, "readEpics: dataset = " << ds.name() << " idx = " << idx);
  
  // This method is only called for non-CTRL data while header may still come from CTRL PV 
  ns_EpicsPvHeader_v0::dataset_data hdr;
  hdr.pvId = pvhdr.pvId();
  hdr.dbrType = pvhdr.dbrType();
  hdr.numElements = pvhdr.numElements();
  
  switch (pvhdr.dbrType()) {
  case Psana::Epics::DBR_CTRL_STRING:
    hdr.dbrType = Psana::Epics::DBR_TIME_STRING;
    break;
  case Psana::Epics::DBR_CTRL_SHORT:
    hdr.dbrType = Psana::Epics::DBR_TIME_SHORT;
    break;
  case Psana::Epics::DBR_CTRL_FLOAT:
    hdr.dbrType = Psana::Epics::DBR_TIME_FLOAT;
    break;
  case Psana::Epics::DBR_CTRL_ENUM:
    hdr.dbrType = Psana::Epics::DBR_TIME_ENUM;
    break;
  case Psana::Epics::DBR_CTRL_CHAR:
    hdr.dbrType = Psana::Epics::DBR_TIME_CHAR;
    break;
  case Psana::Epics::DBR_CTRL_LONG:
    hdr.dbrType = Psana::Epics::DBR_TIME_LONG;
    break;
  case Psana::Epics::DBR_CTRL_DOUBLE: 
    hdr.dbrType = Psana::Epics::DBR_TIME_DOUBLE;
    break;
  default:
    break;
  }

  return ::readEpics(ds, idx, hdr);
}


} // namespace Epics
} // namespace psddl_hdf2psana

namespace {

/**
 *  Read data from specified group and convert them into EPICS object
 */
boost::shared_ptr<Psana::Epics::EpicsPvHeader>
readEpics(const hdf5pp::DataSet& ds, int64_t idx, const ns_EpicsPvHeader_v0::dataset_data& hdr)
{
  MsgLog(logger, debug, "readEpics: dataset = " << ds.name() << " idx = " << idx);
  
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> result;

  switch (hdr.dbrType) {
  case Psana::Epics::DBR_TIME_STRING:
    result = boost::make_shared<EpicsPvTime<dbr_time<const char*> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_SHORT:
    result = boost::make_shared<EpicsPvTime<dbr_time<int16_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_FLOAT:
    result = boost::make_shared<EpicsPvTime<dbr_time<float> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_ENUM:
    result = boost::make_shared<EpicsPvTime<dbr_time<EpicsEnumTag> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_CHAR:
    result = boost::make_shared<EpicsPvTime<dbr_time<uint8_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_LONG:
    result = boost::make_shared<EpicsPvTime<dbr_time<int32_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_TIME_DOUBLE:
    result = boost::make_shared<EpicsPvTime<dbr_time<double> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_STRING:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<const char*> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_SHORT:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<int16_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_FLOAT:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<float> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_ENUM:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<EpicsEnumTag> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_CHAR:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<uint8_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_LONG:
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<int32_t> > >(ds, idx, hdr);
    break;
  case Psana::Epics::DBR_CTRL_DOUBLE: 
    result = boost::make_shared<EpicsPvCtrl<dbr_ctrl<double> > >(ds, idx, hdr);
    break;
  default:
    break;
  }
  
  return result;
}


}
