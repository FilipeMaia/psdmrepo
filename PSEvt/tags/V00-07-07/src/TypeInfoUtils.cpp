#include "PSEvt/TypeInfoUtils.h"

#include <cxxabi.h>
#include <stdlib.h>

using namespace PSEvt;

std::string TypeInfoUtils::typeInfoRealName(const std::type_info *typeInfoPtr) {
    int status;
    char* realname = abi::__cxa_demangle(typeInfoPtr->name(), 0, 0, &status);
    std::string realNameStr(realname);
    free(realname);
    return realNameStr;
}
