@include "psddldata/arraychar.ddl";

@package Arraychar  {

@h5schema DataV1
  [[version(0)]]
{
  @dataset data {
    @attribute numChars;
    @attribute data [[vlen]];
  }
}

} //- @package Arraychar
