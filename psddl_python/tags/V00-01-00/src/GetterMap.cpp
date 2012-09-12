#include <cassert>
#include <stdio.h>
#include <string.h>
#include <psddl_python/GetterMap.h>

namespace psddl_python {

  GetterMap envObjectStoreGetterMap("Psana::EnvObjectStore");
  GetterMap eventGetterMap("Psana::Event");

  void GetterMap::printTables() {
    printf("*** getterMap:\n");
    map<string, Getter*>::const_iterator it;
    for (it = getterMap.begin(); it != getterMap.end(); it++) {
      string typeName = it->first;
      printf("'%s' -> %p\n", it->first.c_str(), it->second);
    }

    printf("*** versionMax:\n");
    map<string, int>::const_iterator vit;
    for (vit = versionMax.begin(); vit != versionMax.end(); vit++) {
      string typeName = vit->first;
      int maxVersion = vit->second;
      int minVersion = versionMin[typeName];
      printf("'%s' -> [%d..%d]\n", typeName.c_str(), minVersion, maxVersion);
    }

    printf("*** versionMin:\n");
    for (vit = versionMin.begin(); vit != versionMin.end(); vit++) {
      string typeName = vit->first;
      int minVersion = vit->second;
      int maxVersion = versionMax[typeName];
      printf("'%s' -> [%d..%d]\n", typeName.c_str(), minVersion, maxVersion);
    }
  }

  void GetterMap::addGetter(Getter* getter) {

    // Add the mapping for the original name (with version).
    // E.g. "BldDataEBeamV0" or "TdcDataV1_Item"
    string typeName(getter->getTypeName());
    getterMap[typeName] = getter;
    const int version = getter->getVersion();
    if (version == -1) {
      return;
    }

    // This getter has a version. See if we find it embedded in the type name.

    static char vpart[64];
    sprintf(vpart, "V%d", version);
    const size_t vpos = typeName.rfind(vpart);
    if (vpos == string::npos) {
      if (version != 0) {
        fprintf(stderr, "%s::addGetter(%s): version is %d but no V%d found in class name\n",
                m_className, typeName.c_str(), version, version);
      }
      return;
    }
    string prefix = typeName.substr(0, vpos).c_str();
    int vpartlen = strlen(vpart);
    string suffix = typeName.substr(vpos + vpartlen);

    string _template = prefix + "V%d" + suffix;

    typeName = prefix + suffix;
    //printf("%s::addGetter generated template '%s' from '%s'\n", m_className, _template.c_str(), typeName.c_str());
    if (templateMap.find(typeName) == templateMap.end()) {
      templateMap[typeName] = _template;
      versionMin[typeName] = version;
      versionMax[typeName] = version;
    } else if (version < versionMin[typeName]) {
      versionMin[typeName] = version;
    } else if (version > versionMax[typeName]) {
      versionMax[typeName] = version;
    }
  }

  // Return template for versionless type name, if one exists.
  const char* GetterMap::getTemplate(const string& typeName, int* pVersionMin, int* pVersionMax) {
    if (templateMap.find(typeName) == templateMap.end()) {
      return NULL;
    }
    const string& _template = templateMap[typeName];
    *pVersionMin = versionMin[typeName];
    *pVersionMax = versionMax[typeName];
    return _template.c_str();
  }

  Getter* GetterMap::getGetter(const string& typeName) {
    if (getterMap.find(typeName) != getterMap.end()) {
      return getterMap[typeName];
    } else {
      return NULL;
    }
  }
}
