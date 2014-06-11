@include "psddldata/alias.ddl";
@include "psddldata/xtc.ddl" [[headers("pdsdata/xtc/Src.hh")]];
@include "psddl_hdf2psana/xtc-h5.ddl" [[headers("psddl_hdf2psana/xtc.h")]];
@package Alias  {


//------------------ SrcAlias ------------------
@h5schema SrcAlias
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute src;
    @attribute aliasName;
  }
}


//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
{
  @dataset config {
    @attribute numSrcAlias;
  }
  @dataset aliases [[method(srcAlias)]];
}
} //- @package Alias
