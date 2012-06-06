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
  static map<int, string> pythonTypeIdMap; // map Python type id to versionless type name
  static map<string, string> pythonTypeNameMap; // map Python type name to versionless type name

  void printTables() {
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
    vit;
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
    printf("adding '%s' -> %p\n", typeName.c_str(), getter);

    string sTypeName;
    const int version = getter->getVersion();
    if (version > 0) {
      static char vpart[64];
      sprintf(vpart, "V%d", version);
      sTypeName = string(typeName);
      const size_t vpos = sTypeName.rfind(vpart);
      typeName = sTypeName.substr(0, vpos).c_str(); // remove version part from typename
    }

    const int typeId = getter->getTypeId();
    if (typeId != -1) {
      const string& pythonTypeName = Pds::TypeId::name(Pds::TypeId::Type(typeId));
      pythonTypeIdMap[typeId] = typeName;
      //printf("adding xtc.TypeId.Type.Id_%s (%d)\n", pythonTypeName.c_str(), typeId);

      pythonTypeNameMap[pythonTypeName] = typeName;
      //printf("adding '%s'\n", pythonTypeName.c_str());
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

  object GenericGetter::get(int typeId, GetMethod* getMethod) {
    printf("!!! getGetterByTypeId(%d)...\n", typeId);
    if (pythonTypeIdMap.find(typeId) == pythonTypeIdMap.end()) {
      printTables();
      printf("getGetterByTypeId(%d): not found\n", typeId);
      return object();
    }
    string& typeName = pythonTypeIdMap[typeId];
    printf("getter->getTypeName()='%s'\n", typeName.c_str());
    return get(typeName, getMethod);
  }

  static string addVersion(string& typeName, int version) {
    char versionedTypeName[typeName.size() + 64];
    sprintf(versionedTypeName, "%sV%d", typeName.c_str(), version);
    return string(versionedTypeName);
  }

  object GenericGetter::get(string& typeName, GetMethod* getMethod) {
    int maxVersion = versionMaxMap[typeName];
    if (maxVersion == 0) {
      // This is not a versioned type name.
      // First, try typeName as a C++ type.
      printf("Looking for class '%s'.\n", typeName.c_str());
      if (typeNameMap.find(typeName) != typeNameMap.end()) {
        GenericGetter* getter = typeNameMap[typeName];
        assert(getter != NULL);
        printf("Found class '%s': %p\n", typeName.c_str(), getter);
        return getMethod->get(getter);
      }
      // If typeName is a python type name, then try again with the corresponding C++ type.
      if (pythonTypeNameMap.find(typeName) != pythonTypeNameMap.end()) {
        string& cppTypeName = pythonTypeNameMap[typeName];
        printf("Trying class '%s' for Python type '%s'.\n", cppTypeName.c_str(), typeName.c_str());
        return get(cppTypeName, getMethod);
      }
      // No getter could be found.
      printTables();
      printf("Could not find C++ class or Python type for '%s'.\n", typeName.c_str());
      return object();
    } else {
      // This is a versioned type name.
      const int minVersion = versionMinMap[typeName];
#if 1
      if (minVersion == maxVersion) {
        string versionedTypeName = addVersion(typeName, minVersion);
        return get(versionedTypeName, getMethod);
      }
#endif
      if (minVersion != maxVersion) {
        printf("Trying classes '%sV%d' through '%sV%d'...\n", typeName.c_str(), minVersion, typeName.c_str(), maxVersion);
      }
      for (int version = maxVersion; version >= minVersion; version--) {
#if 0
        sprintf(versionedTypeName, "%sV%d", typeName.c_str(), version);
#else
        string versionedTypeName = addVersion(typeName, version);
#endif
        printf("Trying class '%s'...\n", versionedTypeName.c_str());
        if (typeNameMap.find(versionedTypeName) != typeNameMap.end()) {
          GenericGetter* getter = typeNameMap[versionedTypeName];
          assert(getter != NULL);
          printf("Found class '%s': %p\n", versionedTypeName.c_str(), getter);
          object result = getMethod->get(getter);
          if (result != object()) {
            printf("Returning result for class '%s'.\n", versionedTypeName.c_str());
            return result;
          }
          printf("get() failed for class '%s'.\n", versionedTypeName.c_str());
        } else {
          printf("Class '%s' was not found.\n", versionedTypeName.c_str());
        }
      }
      printTables();
      printf("None of '%sV%d' through '%sV%d' worked.\n", typeName.c_str(), minVersion, typeName.c_str(), maxVersion);
      return object();
    }
  }
}
