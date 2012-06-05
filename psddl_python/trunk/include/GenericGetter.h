#ifndef PSANA_GENERICGETTER_H
#define PSANA_GENERICGETTER_H

#include <typeinfo>
#include <boost/python.hpp>

namespace Psana {
  using boost::python::api::object;
  using std::string;

  class GetMethod {
  public:
    virtual object get(class GenericGetter* getter) = 0;
    virtual ~GetMethod() {}
  };

  class GenericGetter {
  public:
    virtual const char* getTypeName() = 0; // C++ type name
    virtual int getTypeId() { return -1; } // Python typeId
    virtual int getVersion() { return 0; }
    virtual ~GenericGetter() {}
    static object get(int typeId, GetMethod* getMethod);
    static object get(string& typeName, GetMethod* getMethod);
    static void addGetter(GenericGetter* getter);
  };

}

#endif // PSANA_GENERICGETTER_H
