@package L3T  {


//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_L3TConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  uint32_t _module_id_len -> module_id_len;	/* Length of the module identification string */
  uint32_t _desc_len -> desc_len;	/* Length of the description string */
  char _module_id[@self.module_id_len()] -> module_id;	/* The module identification string */
  char _desc[@self.desc_len()] -> desc;	/* The description string */

  /* Constructor with values for each attribute */
  @init()  [[auto]];

}


//------------------ DataV1 ------------------
@type DataV1
  [[type_id(Id_L3TData, 1)]]
  [[pack(4)]]
{
  uint32_t _accept -> accept;	/* Module trigger decision */

  /* Constructor with values for each attribute */
  @init()  [[auto, inline]];

}
} //- @package L3T
