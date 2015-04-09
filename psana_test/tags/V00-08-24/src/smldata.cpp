#include <stdio.h>
#include "psana_test/smldata.h"
#include <stdexcept>

#define __STDC_FORMAT_MACROS
#include "inttypes.h"

bool isSmallData(const Pds::TypeId &typeId) {
  return (typeId.id() == Pds::TypeId::Id_SmlDataProxy);
}

void parseSmallData(FILE *fout, const Pds::TypeId &typeId, const char *payload) {
  int version = typeId.version();
  switch (version) {
  case 1:
  case 32769:
    const Pds::SmlData::ProxyV1 * smlDataProxy = static_cast<const Pds::SmlData::ProxyV1 *>(static_cast<const void *>(payload));
    int64_t fileOffset = smlDataProxy->fileOffset();
    uint32_t extent = smlDataProxy->extent();
    fprintf(fout, " fileOffset=%" PRId64 " (=0X%8.8" PRIx64 ") extent=%" PRIu32 " (=0X%4.4" PRIx32 ")",
            fileOffset, fileOffset, extent, extent);
    return;
  }
  throw std::runtime_error("parseSmallData - unknown version");
}

