@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/Src.hh")]];
@package Partition  {


//------------------ Segment ------------------
@type Segment
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
  uint32_t _numSegments -> numSegments;	  /* Number of segment definitions */
  Segment _segments[@self.numSegments()] -> segments  [[shape_method(segments_shape)]];	/* Segment configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package Partition
