@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/Src.hh")]];
@package Partition  {


//------------------ Source ------------------
@type Source
  [[value_type]]
  [[pack(4)]]
{
  Pds.Src  _src   -> src;
  uint32_t _group -> group;

  /* Constructor which takes values for every attribute */
  @init() [[auto]];
}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_PartitionConfig, 1)]]
  [[config_type]]
{
  uint64_t _bldMask     -> bldMask;       /* Mask of requested BLD */
  uint32_t _numSources -> numSources;	  /* Number of source definitions */
  Source _sources[@self.numSources()] -> sources  [[shape_method(sources_shape)]];	/* Source configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package Partition
