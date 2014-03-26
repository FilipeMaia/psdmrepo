#ifndef TRANSLATOR_DONOTRANSLATE_H
#define TRANSLATOR_DONOTRANSLATE_H

#include <string>

namespace Translator {

const std::string & doNotTranslateKeyPrefix();
bool hasDoNotTranslatePrefix(const std::string &key, std::string *keyWithPrefixStripped=NULL);

} // namespace

#endif
