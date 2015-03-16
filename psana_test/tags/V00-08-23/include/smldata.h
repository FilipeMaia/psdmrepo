#ifndef PSANA_TEST_SMLDATA
#define PSANA_TEST_SMLDATA

#include "psddl_pds2psana/smldata.ddl.h"

bool isSmallData(const Pds::TypeId &typeId);
void parseSmallData(FILE *fout, const Pds::TypeId &typeId, const char *payload);

#endif
