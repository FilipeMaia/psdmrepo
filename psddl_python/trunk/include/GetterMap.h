#ifndef PSANA_GETTERMAP_H
#define PSANA_GETTERMAP_H

#include <string>
#include <map>
#include <pdsdata/xtc/Src.hh>
#include <psddl_python/Getter.h>

namespace psddl_python {
  using std::map;
  using std::string;
  using Pds::Src;

  class GetterMap {
  private:
    const char* m_className;
    map<string, int> versionMin; // map versionless type name to minimum version
    map<string, int> versionMax; // map versionless type name to maxium version
    map<string, Getter*> getterMap;  // map C++ type name (version NOT removed) to getter
    map<string, string> templateMap; // map C++ type name (version IS removed) to sprintf format string
    void printTables();
  public:
    GetterMap(const char* className) : m_className(className) {}
    void addGetter(Getter* getter);
    Getter* getGetter(const string& typeName);
    const char* getTemplate(const string& typeName, int* versionMin, int* versionMax);
  };

  extern GetterMap envObjectStoreGetterMap;
  extern GetterMap eventGetterMap;
}

#endif // PSANA_GETTERMAP_H
