#ifndef PSANA_GETTER_H
#define PSANA_GETTER_H

namespace Psana {
  class Getter {
  public:
    virtual const char* getTypeName() = 0; // C++ type name
    virtual int getVersion() { return -1; }
    virtual ~Getter() {}
  };
}

#endif // PSANA_GETTER_H
