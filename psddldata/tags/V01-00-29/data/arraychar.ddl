@package Arraychar  {

@type DataV1
  [[type_id(Id_Arraychar, 1)]]
  [[pack(4)]]
{
  uint64_t _iNumChars -> numChars;
  uint8_t  _data[@self._iNumChars] -> data;

  /* Constructor with values for scalar attributes */
  @init(iNumChars -> _iNumChars)  [[inline]];
}

} //- @package Arraychar
