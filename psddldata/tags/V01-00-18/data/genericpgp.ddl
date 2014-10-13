@package GenericPgp  {

//------------------ CDimension ------------------
@type CDimension
  [[pack(4)]]
{
  uint32_t _rows     -> rows;
  uint32_t _columns  -> columns;

  /* Constructor with value for each attribute */
  @init()  [[auto, inline]];
}

//------------------ CRegister ------------------
@type CRegister
  [[value_type]]
  [[pack(4)]]
{
  @enum Action (uint8_t) {
    RegisterRead  =0,  // Read and store
    RegisterWrite =1,  // Write
    RegisterWriteA=2,  // Write and wait for ack
    RegisterVerify=3,  // Read and verify
    Spin          =4,  // Spin lock
    Usleep        =5,  // Sleep
    Flush         =6,  // Flush pending reads
  }
  uint32_t _Action {
    /* Configuration action */
    Action   _action:8 -> action;
    /* Size of register access (in uint32_t's) */
    uint32_t _datasize:24 -> datasize;
  }
  /* Register access address */
  uint32_t _address  -> address;
  /* Payload offset */
  uint32_t _offset   -> offset;
  /* Register value mask */
  uint32_t _mask     -> mask;

  /* Constructor with value for each attribute */
  @init()  [[auto, inline]];
}

//------------------ CStream ------------------
@type CStream
  [[value_type]]
  [[pack(4)]]
{
  /* PGP virtual channel ID */
  uint32_t _pgp_channel   -> pgp_channel;

  /* Event data type ID */
  uint32_t _data_type     -> data_type;

  /* Configuration data type ID */
  uint32_t _config_type   -> config_type;

  /* Location of configuration data */
  uint32_t _config_offset -> config_offset;

  /* Constructor with value for each attribute */
  @init()  [[auto, inline]];
}  

//------------------ ConfigV1 ------------------
@type ConfigV1
  [[type_id(Id_GenericPgpConfig, 1)]]
  [[config_type]]
  [[pack(4)]]
{
  /* Serial number identifying the array */
  uint32_t   _id  -> id;

  /* Dimensions of the frame data from the array */
  CDimension _frame_dim -> frame_dim;

  /* Dimensions of the auxillary data from the array */
  CDimension _aux_dim -> aux_dim;

  /* Dimensions of the environmental data from the array */
  CDimension _env_dim -> env_dim;

  /* Number of registers in the sequence array */
  uint32_t   _number_of_registers     -> number_of_registers;

  /* Number of (sub)sequences of register operations in the array */
  uint32_t   _number_of_sequences     -> number_of_sequences;

  uint32_t   _number_of_streams       -> number_of_streams;

  uint32_t   _payload_size            -> payload_size;

  uint32_t   _pixel_settings[@self.frame_dim().rows()][@self.frame_dim().columns()] -> pixel_settings;

  /* Lengths of (sub)sequence register operations in the array */
  uint32_t   _sequence_length[@self.number_of_sequences()] -> sequence_length;

  /* Register Operations */
  CRegister  _sequence[@self.number_of_registers()] -> sequence;

  /* Stream readout configuration */
  CStream    _stream[@self.number_of_streams()] -> stream;

  /* Stream and Register Data */
  uint32_t   _payload[@self.payload_size()] -> payload;

  /* Number of rows in a readout unit */
  uint32_t numberOfRows()  [[inline]]
  [[language("C++")]] @{ return @self.frame_dim().rows(); @}

  /* --- Epix.ElementV1 interface --- */
  /* Number of columns in a readout unit */
  uint32_t numberOfColumns()  [[inline]]
  [[language("C++")]] @{ return @self.frame_dim().columns(); @}

  /* Number of rows in the auxillary data */
  uint32_t lastRowExclusions()  [[inline]]
  [[language("C++")]] @{ return @self.aux_dim().rows(); @}

  /* Number of elements in environmental data */
  uint32_t numberOfAsics()  [[inline]]
  [[language("C++")]] @{ return @self.env_dim().columns(); @}
  /* --- End of Epix.ElementV1 interface --- */


  /* Constructor which takes values for every attribute */
  @init()  [[auto]];

  /* Constructor which takes values necessary for size calculations */
  @init(arg__frame_dim               -> _frame_dim, 
	arg__number_of_registers     -> _number_of_registers,
	arg__number_of_sequences     -> _number_of_sequences,
        arg__number_of_streams       -> _number_of_streams,
	arg__payload_size            -> _payload_size) [[inline]];
}

}