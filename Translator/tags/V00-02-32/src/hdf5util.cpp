#include <string>
#include <sstream>
#include <cmath>

#include "MsgLogger/MsgLogger.h"
#include "Translator/hdf5util.h"
#include "hdf5/hdf5_hl.h"
#include "ErrSvc/Issue.h"

using namespace std;

namespace {
  string logger(string addto="") {
    const string base = "hdf5util";
    return (addto.size()>0 ? base + string(".") + addto : base);
  }
} // local namespace

void Translator::hdf5util::addAttribute_uint64(hid_t hid, const char * name, uint64_t val) {
  const string addTo = "addAttribute_uint64";
  hsize_t dims = 1;
  hid_t dspace = H5Screate_simple(1,&dims,NULL);
  hid_t attr = H5Acreate2(hid, name, H5T_NATIVE_UINT64, dspace, H5P_DEFAULT, H5P_DEFAULT);
  if (dspace<0 or attr<0) {
    MsgLog(logger(addTo), fatal, 
           "bad dspace (=" << dspace << ") or attr id=("
           << attr << ")");
  }
  herr_t res = H5Awrite(attr, H5T_NATIVE_UINT64, &val);
  if (res<0) MsgLog(logger(addTo), fatal, "H5Awrite call failed");
  res = H5Aclose(attr);
  res = std::min(res,H5Sclose(dspace));
  if (res<0) MsgLog(logger("addAttribute_uint64"), fatal, "failed to close attr or dpsace");
}
                 

// this function is based on hdf5pp::Group::name()
std::string Translator::hdf5util::objectName(hid_t obj)
{
  const int maxsize = 255;
  char buf[maxsize+1];

  // first try with the fixed buffer size
  ssize_t size = H5Iget_name(obj, buf, maxsize+1);
  if (size < 0) {
    ostringstream msg;
    msg << "objectName: H5Iget_name call failed for hid_t=" << obj;
    throw ErrSvc::Issue( ERR_LOC, msg.str());
  }
  if (size == 0) {
    // name is not known
    return std::string();
  }
  if (size <= maxsize) {
    // name has fit into buffer
    return buf;
  }

  // another try with dynamically allocated buffer
  char* dbuf = new char[size+1];
  H5Iget_name(obj, dbuf, size+1);
  std::string res(dbuf);
  delete [] dbuf;
  return res;
}

// This function is based on hdf5pp::Type << operator
// dumps type information in HDF5 DDL format.
std::string Translator::hdf5util::type2str(hid_t id) {
  std::string out;
  size_t len = 0;
  herr_t err = H5LTdtype_to_text(id, 0, H5LT_DDL, &len);
  if (err >= 0) {
    char* buf = new char[len];
    err = H5LTdtype_to_text(id, buf, H5LT_DDL, &len);
    if (err >= 0) out = buf;
    delete [] buf;
  }
  return out;
 }
