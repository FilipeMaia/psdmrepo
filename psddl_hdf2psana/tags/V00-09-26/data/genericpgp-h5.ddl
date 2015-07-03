@include "psddldata/genericpgp.ddl";
@package GenericPgp  {


//------------------ CDimension ------------------
@h5schema CDimension
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute rows [[method(rows)]]; 
    @attribute columns [[method(columns)]]; 
  }
} 

//------------------ CRegister ------------------
@h5schema CRegister
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute action [[method(action)]];
    @attribute datasize [[method(datasize)]];
    @attribute address [[method(address)]];
    @attribute offset [[method(offset)]];
    @attribute mask [[method(mask)]];
  }
}

//------------------ CStream ------------------
@h5schema CStream
  [[version(0)]]
  [[embedded]]
{
  @dataset data {
    @attribute pgp_channel [[method(pgp_channel)]];
    @attribute data_type [[method(data_type)]];
    @attribute config_type [[method(config_type)]];
    @attribute config_offset [[method(config_offset)]];
  }
}

//------------------ ConfigV1 ------------------
@h5schema ConfigV1
  [[version(0)]]
  [[default]]
{
}

} //- @GenericPgp
