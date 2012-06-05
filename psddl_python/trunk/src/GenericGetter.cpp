#include <string>
#include <map>
#include <vector>
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

  void GenericGetter::addGetter(GenericGetter* getter) {
    const char* typeName = getter->getTypeName();
    typeNameMap[typeName] = getter;
    printf("adding %s\n", typeName);

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
      printf("adding xtc.TypeId.Type.Id_%s (%d)\n", pythonTypeName.c_str(), typeId);

      pythonTypeNameMap[pythonTypeName] = typeName;
      printf("adding %s\n", pythonTypeName.c_str());
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

    string& typeName = pythonTypeIdMap[typeId];
    if (typeName == "") {
      printf("getGetterByTypeId(%d): not found\n", typeId);
      return object();
    }
    typeName = pythonTypeIdMap[typeId];
    printf("getter->getTypeName()=%s\n", typeName.c_str());
    return get(typeName, getMethod);
  }

  static object call(GetMethod* getMethod, GenericGetter* getter) {
    if (! getMethod) {
      printf("!!! GenericGetter::call(): getMethod is null!\n");
      exit(1);
    }
    return getMethod->get(getter);
  }

  object GenericGetter::get(string& typeName, GetMethod* getMethod) {
    int maxVersion = versionMaxMap[typeName];
    if (maxVersion == 0) {
      // This is not a versioned type name.
      // First, try typeName as a C++ type.
      GenericGetter* getter = typeNameMap[typeName];
      if (getter) {
        printf("Found unversioned getter for C++ type %s.\n", typeName.c_str());
        return call(getMethod, getter);
      }
      // Next, try typeName as a python type -- look up the corresponding C++ type.
      string& cppTypeName = pythonTypeNameMap[typeName];
      if (cppTypeName == "") {
        printf("Could find neither getter nor C++ type for %s.\n", typeName.c_str());
        return object();
      }
      string& pythonTypeName = typeName;
      typeName = cppTypeName;
      printf("Trying C++ type %s for python type %s.\n", typeName.c_str(), pythonTypeName.c_str());
      getter = typeNameMap[typeName];
      if (getter) {
        printf("Found getter for C++ type %s (Python type %s).\n", typeName.c_str(), pythonTypeName.c_str());
        return call(getMethod, getter);
      }
      // update maxVersion for translated (C++) typeName.
      maxVersion = versionMaxMap[typeName];
      if (maxVersion == 0) {
        printf("Nothing found for unversioned type %s.\n", typeName.c_str());
        return object();
      }
    }

    // This is a versioned type name.
    char versionedTypeName[typeName.size() + 64];
    const int minVersion = versionMinMap[typeName];
    printf("Trying versioned C++ type %s with versions %d..%d\n", typeName.c_str(), minVersion, maxVersion);
    for (int version = maxVersion; version >= minVersion; version--) {
      sprintf(versionedTypeName, "%sV%d", typeName.c_str(), version);
      printf("Trying %s...\n", versionedTypeName);
      GenericGetter* getter = typeNameMap[versionedTypeName];
      if (getter) {
        printf("Found getter for versioned C++ type %s.\n", versionedTypeName);
        object result = call(getMethod, getter);
        if (result != object()) {
          printf("Got result for versioned C++ type %s.\n", versionedTypeName);
          return result;
        }
        printf("No result was found when calling getMethod for %s.\n", versionedTypeName);
      } else {
        printf("No getter was found for %s.\n", versionedTypeName);
      }
    }

    printf("Nothing found for versioned type %s.\n", typeName.c_str());
    return object();
  }
}
