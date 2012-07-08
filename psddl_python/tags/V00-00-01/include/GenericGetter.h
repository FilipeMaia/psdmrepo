// This is not a conventional include file, but rather a template
// that is included by EventGetter.cpp and EnvObjectStore.cpp.

#include <sstream>
#include "PSEvt/EventKey.h" // operator<<(Pds::Src) defined here

namespace Psana {
  static string toString(const Src& src) {
    std::ostringstream stream;
    stream << src;
    return stream.str();
  }

  static void printGetResult(const char* typeName, const char* foundTypeName, const Src* foundSrc) {
    printf("get(%s) -> %s -> %s\n",
           typeName, foundTypeName, toString(*foundSrc).c_str());
  }

  object CLASS::get(ARGS_DECL) {
    Src src;
    if (foundSrc == NULL) {
      foundSrc = &src;
    }
    CLASS* getter = (CLASS*) MAP.getGetter(typeName);
    if (getter) {
      object result(getter->get(ARGS));
      //printGetResult(typeName.c_str(), typeName.c_str(), foundSrc);
      return result;
    }
    int versionMin, versionMax;
    // A template is of the form PSana::Fli::FrameV1
    const char* _template = MAP.getTemplate(typeName, &versionMin, &versionMax);
    if (! _template) {
      printf("%s: no getter exists for class '%s'\n", CLASS_NAME, typeName.c_str());
      return object();
    }
    for (int version = versionMin; version <= versionMax; version++) {
      char vTypeName[128];
      sprintf(vTypeName, _template, version);
      getter = (CLASS*) MAP.getGetter(vTypeName);
      if (getter) {
        object result(getter->get(ARGS));
        if (result != object()) {
          //printGetResult(typeName.c_str(), vTypeName, foundSrc);
          return result;
        }
        //printf("%s: tried %s but it returned null\n", CLASS_NAME, vTypeName);
      } else {
        printf("%s: no getter exists for class '%s'\n", CLASS_NAME, typeName.c_str());
      }
    }
    //printf("%s: no object found for %s\n", CLASS_NAME, _template);
    return object();
  }
}
