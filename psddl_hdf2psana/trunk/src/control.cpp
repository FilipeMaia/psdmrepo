/**
 *  Few methods missing from generated code
 */

#include "psddl_hdf2psana/control.ddl.h"

namespace psddl_hdf2psana {
namespace ControlData {

uint32_t
ConfigV2_v0::npvLabels() const
{
  return pvLabels().size();
}

uint32_t
ConfigV2_v1::npvMonitors() const
{
  return pvMonitors().size();
}

uint32_t
ConfigV3_v0::npvLabels() const
{
  return pvLabels().size();
}

} // namespace ControlData
} // namespace psddl_hdf2psana
