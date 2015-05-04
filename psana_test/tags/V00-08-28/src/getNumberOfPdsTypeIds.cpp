#include "pdsdata/xtc/TypeId.hh"

extern "C" {
int getNumberOfPdsTypeIds();
}

int getNumberOfPdsTypeIds() {
  const static int numberOf = Pds::TypeId::NumberOf;
  return numberOf;
}
