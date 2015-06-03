@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/TypeId.hh")]];

@package SmlData {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_SmlDataConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _sizeThreshold -> sizeThreshold;

  /* Construct from all attributes */
  @init()  [[auto, inline]];

}

@type OrigDgramOffsetV1
  [[type_id(Id_SmlDataOrigDgramOffset, 1)]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  int64_t    _fileOffset -> fileOffset;
  uint32_t   _extent     -> extent;

  /* Construct from all attributes */
  @init()  [[auto, inline]];
}


@type ProxyV1
  [[type_id(Id_SmlDataProxy, 1)]]
  [[pack(4)]]
  [[config(ConfigV1)]]
{
  int64_t    _fileOffset  -> fileOffset;
  Pds.TypeId _type        -> type;
  uint32_t   _extent      -> extent;

  /* Construct from all attributes */
  @init()  [[auto, inline]];
}

} //- @package Pds
