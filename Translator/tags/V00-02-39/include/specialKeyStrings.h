#ifndef TRANSLATOR_SPECIALKEYSTRINGS_H
#define TRANSLATOR_SPECIALKEYSTRINGS_H

#include <string>

namespace Translator {

/// returns "do_not_translate"
const std::string & doNotTranslatePrefix();

/// returns true if "do_not_translate" starts key.
/// optionally, will return a string that has "do_not_translate", 
/// "do_not_translate:" or "do_not_translate_" stripped from the
/// input key. This is returned through the output argument - 
/// keyWithPrefixStripped
bool hasDoNotTranslatePrefix(const std::string &key, std::string *keyWithPrefixStripped=NULL);

/// returns translate_vlen
const std::string & ndarrayVlenPrefix();


/// returns true if "translate_vlen" starts key.
/// optionally, will return a string that has "translate_vlen", 
/// "translate_vlen:" or "translate_vlen_" stripped from the
/// input key. This is returned through the output argument - 
/// keyWithPrefixStripped
bool hasNDArrayVlenPrefix(const std::string &key, std::string *keyWithPrefixStripped=NULL);

} // namespace

#endif
