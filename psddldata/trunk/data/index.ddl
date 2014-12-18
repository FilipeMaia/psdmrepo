@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/TypeId.hh")]];

@package Index {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_IndexConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _sizeThreshold -> sizeThreshold;

  /* Construct from all attributes */
  @init()  [[auto, inline]];

}

@type TagV1
  [[type_id(Id_IndexTag, 1)]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  uint64_t   _fileOffset -> fileOffset;
  uint32_t   _extent     -> extent;

  /* Construct from all attributes */
  @init()  [[auto, inline]];
}


@type ProxyV1
  [[type_id(Id_IndexProxy, 1)]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  Pds.TypeId _type        -> type;
  uint32_t   _dgramOffset -> dgramOffset;
  uint32_t   _extent      -> extent;

  /* Construct from all attributes */
  @init()  [[auto, inline]];
}

} //- @package Pds
