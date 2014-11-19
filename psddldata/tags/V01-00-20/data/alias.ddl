@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/Src.hh")]];
@package Alias  {


//------------------ SrcAlias ------------------
@type SrcAlias
  [[value_type]]
  [[pack(4)]]
{
  @const int32_t AliasNameMax = 31;

  Pds.Src _src -> src;	/* The src identifier */
  char _aliasName[AliasNameMax] -> aliasName;	/* Alias name for src identifier */
  char _pad1;

  uint8_t operator <(SrcAlias other)
  [[language("C++")]] @{ return (strcmp(@self._aliasName,other._aliasName)<0); @}

  uint8_t operator ==(SrcAlias other)
  [[language("C++")]] @{ return (strcmp(@self._aliasName,other._aliasName)==0); @}

  /* Full constructor */
  @init()
    _pad1(0)  [[auto]];

}


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_AliasConfig, 1)]]
  [[config_type]]
{
  uint32_t _numSrcAlias -> numSrcAlias;	/* Number of alias definitions */
  SrcAlias _srcAlias[@self.numSrcAlias()] -> srcAlias  [[shape_method(srcAlias_shape)]];	/* SrcAlias configuration objects */

  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

}
} //- @package Alias
