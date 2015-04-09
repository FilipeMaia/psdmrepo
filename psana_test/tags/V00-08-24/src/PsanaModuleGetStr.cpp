#include <stdexcept>
#include "psana/Module.h"

/**
 * Testing module that expects a std::string with key 'key' to be in the
 * Event during beginJob and event. If it is not there, it throws an 
 * exception. If it is there it prints a message with the retrieved string.
 * 
 * To be used with the Python module PsanaModulePutStr which puts the string in.
 */
class PsanaModuleGetStr : public Module {
public:
  PsanaModuleGetStr(const std::string &name) : Module(name) {}
  void beginJob(Event &evt, Env &env) {
    boost::shared_ptr<std::string> pyStr = evt.get("key");
    if (not pyStr) throw std::runtime_error("Did not get std::string with key='key' in beginJob");
    MsgLog(name(), info, "received string from key 'key': " << *pyStr);
    
  };
  void event(Event &evt, Env &env) {
    boost::shared_ptr<std::string> pyStr = evt.get("key");
    if (not pyStr) throw std::runtime_error("Did not get std::string with key='key' in beginJob");
    MsgLog(name(), info, "received string from key 'key': " << *pyStr);
  };
};

PSANA_MODULE_FACTORY(PsanaModuleGetStr);
