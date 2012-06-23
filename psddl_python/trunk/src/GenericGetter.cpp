#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <psddl_python/GenericGetter.h>
#include <pdsdata/xtc/TypeId.hh>

using std::string;
using std::map;
using std::type_info;
using std::vector;

namespace Psana {
  static map<string, int> versionMinMap; // map versionless type name to minimum version
  static map<string, int> versionMaxMap; // map versionless type name to maxium version
  static map<string, GenericGetter*> typeNameMap; // map C++ type name (version NOT removed) to getter

  static void printTables() {
    printf("*** typeNameMap:\n");
    map<string, GenericGetter*>::const_iterator it;
    for (it = typeNameMap.begin(); it != typeNameMap.end(); it++) {
      string typeName = it->first;
      printf("'%s' -> %p\n", it->first.c_str(), it->second);
    }

    printf("*** versionMaxMap:\n");
    map<string, int>::const_iterator vit;
    for (vit = versionMaxMap.begin(); vit != versionMaxMap.end(); vit++) {
      string typeName = vit->first;
      int maxVersion = vit->second;
      int minVersion = versionMinMap[typeName];
      printf("'%s' -> [%d..%d]\n", typeName.c_str(), minVersion, maxVersion);
    }

    printf("*** versionMinMap:\n");
    for (vit = versionMinMap.begin(); vit != versionMinMap.end(); vit++) {
      string typeName = vit->first;
      int minVersion = vit->second;
      int maxVersion = versionMaxMap[typeName];
      printf("'%s' -> [%d..%d]\n", typeName.c_str(), minVersion, maxVersion);
    }
  }

  void GenericGetter::addGetter(GenericGetter* getter) {
    string typeName(getter->getTypeName());
    typeNameMap[typeName] = getter;
    //printf("adding '%s' -> %p\n", typeName.c_str(), getter);

    string sTypeName;
    const int version = getter->getVersion();
    if (version > 0) {
      static char vpart[64];
      sprintf(vpart, "V%d", version);
      sTypeName = string(typeName);
      const size_t vpos = sTypeName.rfind(vpart);
      typeName = sTypeName.substr(0, vpos).c_str(); // remove version part from typename
    }

    if (version > 0) {
      int min = versionMinMap[typeName];
      if (min == 0 || min > version) {
        versionMinMap[typeName] = version;
      }
      int max = versionMaxMap[typeName];
      if (max == 0 || max < version) {
        versionMaxMap[typeName] = version;
      }
    }
  }

  static string addVersion(string& typeName, int version) {
    char versionedTypeName[typeName.size() + 64];
    sprintf(versionedTypeName, "%sV%d", typeName.c_str(), version);
    return string(versionedTypeName);
  }

  object GenericGetter::get(string& typeName, GetMethod* getMethod) {
    if (typeName == "") {
      printf("GenericGetter:get(): no typeName!\n");
      PyErr_Print();
      return object();
    }
    int maxVersion = versionMaxMap[typeName];
    if (maxVersion == 0) {
      // This is not a versioned type name.
      // First, try typeName as a C++ type.
      //printf("Looking for class '%s'.\n", typeName.c_str());
      if (typeNameMap.find(typeName) != typeNameMap.end()) {
        GenericGetter* getter = typeNameMap[typeName];
        assert(getter != NULL);
        //printf("Found class '%s': %p\n", typeName.c_str(), getter);
        return getMethod->get(getter);
      }
      // No getter could be found.
      //printTables();
      char buf[1024];
      sprintf(buf, "Could not find class '%s'.\n", typeName.c_str());
      PyErr_SetString(PyExc_LookupError, buf);
      PyErr_Print();
      return object();
    } else {
      // This is a versioned type name.
      const int minVersion = versionMinMap[typeName];
      if (minVersion == maxVersion) {
        // No need to iterate, as there is only possible versioned type name.
        string versionedTypeName = addVersion(typeName, minVersion);
        return get(versionedTypeName, getMethod);
      }
      int testCount = 0;
      for (int version = maxVersion; version >= minVersion; version--) {
        string versionedTypeName = addVersion(typeName, version);
        //printf("Trying class '%s'...\n", versionedTypeName.c_str());
        if (typeNameMap.find(versionedTypeName) != typeNameMap.end()) {
          GenericGetter* getter = typeNameMap[versionedTypeName];
          assert(getter != NULL);
          //printf("Found class '%s': %p\n", versionedTypeName.c_str(), getter);
          object result = getMethod->get(getter);
          if (result != object()) {
            //printf("Returning result for class '%s'.\n", versionedTypeName.c_str());
            return result;
          }
          //printf("get() failed for class '%s'.\n", versionedTypeName.c_str());
          testCount++;
        } else {
          //printf("Class '%s' was not found.\n", versionedTypeName.c_str());
        }
      }
      char buf[1024];
      if (testCount == 0) {
        sprintf(buf, "None of '%s' through '%s' could be found.\n", 
               addVersion(typeName, minVersion).c_str(),
               addVersion(typeName, maxVersion).c_str());
      } else {
        sprintf(buf, "None of '%s' through '%s' returned a value.\n", 
               addVersion(typeName, minVersion).c_str(),
               addVersion(typeName, maxVersion).c_str());
      }
      PyErr_SetString(PyExc_LookupError, buf);
      PyErr_Print();
      return object();
    }
  }
}
