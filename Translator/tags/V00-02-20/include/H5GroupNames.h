#ifndef TRANSLATOR_H5GROUPNAMES_H
#define TRANSLATOR_H5GROUPNAMES_H

#include <string>
#include <typeinfo>
#include "pdsdata/xtc/Src.hh"
#include "Translator/TypeAliases.h"
#include "hdf5pp/Group.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief transforms C++ Psana types and src locations into hdf5 group names.
 *
 *  Also returns true if a C++ type is a NDArray, requires the set of 
 *  NDArray types recognized in the system to be passed in for this.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class H5GroupNames {
 public:
  H5GroupNames(const std::string & calibratedKey, const TypeAliases::TypeInfoSet & ndarrays);
  /**
   * @brief returns H5 group name for a C++ type, may be modified by key
   *
   * This method returns the string used for the H5 group for data of this
   * type. The is Generally based on the C++ type name, however modifications
   * are made to be backward compatible with o2o-translate and to simplify some
   * names - in particular ndarrays. For example ndarray< int, 2> will return
   * ndarray_int32_2. If a key parameter is passed in, it may modify the type. 
   * Currently the only modification is that ndarrays with the translate_vlen
   * prefix will have _vlen appeneded to the group name. The longest ndarray
   * group name is ndarray_const_float128_vlen
   *
   * @param[in] typeInfoPtr type of object for the group
   * @param[in] key string, will be checked for vlen if type is ndarray
   * @return a non empty string name for the h5 group
   */
  std::string nameForType(const std::type_info *typeInfoPtr, const std::string &key);

  /**
   * @brief adds h5 attributes to group appropriate for type.
   *
   * all types will get the attribute:
   *
   * uint8_t _ndarray
   *
   * If the type is an ndarray (the attribute is set to 1), 
   * then the following additional attributes will be
   * added:
   *
   * uint32_t _ndarrayDim
   * const char * _ndarrayElemType   one of int, uint or float
   * uint32_t _ndarraySizeBytes
   * uint8_t _ndarrayConstElem
   *
   * @param[in] group the hdf5 group to modify and add attributes to
   * @param[in] typeInfoPtr type of object for the group
   * @param[in] key full key from event key, to determin if ndarray is vlen
   */
  void addTypeAttributes(hdf5pp::Group group, const std::type_info *typeInfoPtr, const std::string & key);
  /**
   * @brief returns h5 group name for a Pds::Src
   *
   * This method returns the string used for the H5 group for data from the
   * given Pds::Src. This is most often the same as what the Psana Module
   * EventKeys prints, however adjustments are made for some sources to 
   * produce names appropriate for h5 files. In particular:
   *   @li no spaces or asterick's are returned.
   *   @li any src where the level is Control returns "Control"
   *   @li any src where the level is Event return "Event"
   *   @li the special values noSource and anySource return "noSrc"
   *
   * @param[in] src the Pds::Src for the group
   * @return non empty string name for the h5 group
   */
  std::string nameForSrc(const Pds::Src &src);

  /**
   * @brief returns h5 group name and cleaned key for a Pds::Src and key
   *
   * The h5group name for a src and key is described by a number of cases:
   *  no key string:  a string representation of the pds source
   *                  This most always agrees with what EventKeys prints.
   *  a key string:   Generally, this will be  a __ separated concatenation 
   *                  of the string representation of the src, and the key
   *                    
   * If the source is the special Psana values for no source or any source, 
   * the string 'noSrc' is used for the source.
   *
   * The key string is first checked to see if it is the special key for
   * calibrated data. If so, no key string is used, only the source for the
   * group name and an empty string is returned for the second argument in the
   * pair. 
   * The key is then checked for special key strings (do_not_translate and 
   * translate_vlen) If these are present they are stripped from the key.
   * The key is then checked for characters bad for h5 group names,
   * such as a / this is replaced with _ 
   * 
   * The cleaned key string, striped of special prefixes, is used in the group name.
   *
   * @param[in] src the Pds::Src for the group 
   * @param[in] key the key string for the group
   * @return a pair with the h5 group name first, and the possibly modified
   *         key used to form this name (will be empty for calibrated).
   */
  std::pair<std::string, std::string> nameForSrcKey(const Pds::Src &src, 
                                                    const std::string &key);

  /**
   * @brief returns true if the type is one of the known NDArrays.
   *
   * @param[in] typeInfoPtr the C++ type.
   * @return true if it is a known NDArray.
   */
  bool isNDArray(const std::type_info *typeInfoPtr) { 
    return m_ndarrays.find(typeInfoPtr) != m_ndarrays.end(); 
  }

  /**
   * @brief returns the calibration key.
   *
   * This is the special key that is is ignored when forming the srcKey Group name.
   */
  const std::string &calibratedKey() { return m_calibratedKey; }
 private:
  const std::string m_calibratedKey;
  const TypeAliases::TypeInfoSet m_ndarrays;
}; // class H5GroupNames

} // namespace

#endif
