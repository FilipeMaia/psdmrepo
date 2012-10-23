#define CLASS EventGetter
#define CLASS_NAME "EventGetter"
#define MAP eventGetterMap
#define ARGS event, source, key, foundSrc
#define ARGS_DECL string typeName, Event& event, Source& source, const std::string& key, Src* foundSrc

#include <psddl_python/EventGetter.h>
#include <psddl_python/GenericGetter.h>
