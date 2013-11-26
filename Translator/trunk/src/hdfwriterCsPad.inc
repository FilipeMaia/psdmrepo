 // Redo writer for CsPad::DataV2
template <>
class HdfWriter<Psana::CsPad::DataV2> : public HdfWriterBase {
public:
  HdfWriter<Psana::CsPad::DataV2>() : m_typeArrayDim4uint16(-1), 
                                      m_typeQuad(-1), 
                                      m_typeElement(-1),
                                      m_typeData(-1),
                                      m_typeCommon(-1)
  {}

  void make_datasets(DataTypeLoc dataTypeLoc, hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, PSEnv::Env & env,
                     bool shuffle, int deflate,
                     boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy);

  void store(DataTypeLoc dataTypeLoc, 
             hdf5pp::Group & srcGroup, 
             const PSEvt::EventKey & eventKey, 
             PSEvt::Event & evt, 
             PSEnv::Env & env);

  void store_at(DataTypeLoc dataTypeLoc, 
                long index, hdf5pp::Group & srcGroup, 
                const PSEvt::EventKey & eventKey, 
                PSEvt::Event & evt, 
                PSEnv::Env & env);

  void append(DataTypeLoc dataTypeLoc,
              hdf5pp::Group & srcGroup, const PSEvt::EventKey & eventKey, 
              PSEvt::Event & evt, PSEnv::Env & env);

  void addBlank(hdf5pp::Group & group);

private:
  Translator::DataSetCreationProperties m_dataSetCreationProperties;
  Translator::HdfWriterGeneric m_writer;

  hid_t m_typeArrayDim4uint16, m_typeQuad, m_typeElement, m_typeData, m_typeCommon;

  size_t m_elementDatasetIndex, m_dataDatasetIndex, m_commonModeDatasetIndex;

  struct Element {
    uint32_t virtual_channel;
    uint32_t lane;
    uint32_t tid;
    uint32_t acq_count;
    uint32_t op_code;
    uint32_t quad;
    uint32_t seq_count;
    uint32_t ticks;
    uint32_t fiducials;
    uint32_t frame_type;
    uint32_t sectionMask;
    uint16_t sb_temp[4];
  };
};

void HdfWriter<Psana::CsPad::DataV2>::make_datasets(DataTypeLoc dataTypeLoc, 
                                                    hdf5pp::Group & srcGroup, 
                                                    const PSEvt::EventKey & eventKey, 
                                                    PSEvt::Event & evt, PSEnv::Env & env,
                                                    bool shuffle, int deflate,
                                                    boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy)
{
  checkType<Psana::CsPad::DataV2>(eventKey, "HdfWriter");
  boost::shared_ptr<Psana::CsPad::DataV2> ptr;
  if (dataTypeLoc == inEvent) ptr = evt.get(eventKey.src(), eventKey.key()); 
  else if (dataTypeLoc == inConfigStore) ptr = env.configStore().get(eventKey.src());
  
  m_dataSetCreationProperties = DataSetCreationProperties(chunkPolicy,shuffle, deflate);
  
  // make type for an element
  
  // first type for a quad
  hsize_t dims[1] = {4};
  m_typeArrayDim4uint16 = H5Tarray_create2(H5T_NATIVE_UINT16, 1, dims);
  m_typeQuad = H5Tcreate(H5T_COMPOUND, sizeof(Element));
  H5Tinsert(m_typeQuad, "virtual_channel", offsetof(Element, virtual_channel), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "lane", offsetof(Element, lane), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "tid", offsetof(Element, tid), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "acq_count", offsetof(Element, acq_count), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "op_code", offsetof(Element, op_code), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "quad", offsetof(Element, quad), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "seq_count", offsetof(Element, seq_count), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "ticks", offsetof(Element, ticks), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "fiducials", offsetof(Element, fiducials), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "frame_type", offsetof(Element, frame_type), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "sectionMask", offsetof(Element, sectionMask), H5T_NATIVE_UINT32);
  H5Tinsert(m_typeQuad, "sb_temp", offsetof(Element, sb_temp), m_typeArrayDim4uint16);
  
  const unsigned nquads = ptr->quads_shape()[0];
  unsigned nsect = 0;
  for (unsigned q = 0; q != nquads; ++ q) {
    nsect += ptr->quads(q).data().shape()[0];
  }
  
  // now element type and element dataset
  const hsize_t quadDims = nquads;
  m_typeElement = H5Tarray_create2(m_typeQuad, 1, &quadDims);
  m_elementDatasetIndex = m_writer.createUnlimitedSizeDataset(srcGroup.id(), "element", m_typeElement, m_dataSetCreationProperties);
  
  // now for data
  hsize_t dataDims[3] = {nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2};
  m_typeData = H5Tarray_create2(H5T_NATIVE_UINT16, 3, dataDims);
  m_dataDatasetIndex = m_writer.createUnlimitedSizeDataset(srcGroup.id(), "data", m_typeData, m_dataSetCreationProperties);
  
  // now for common mode
  hsize_t dimCommon[] = { nsect };
  m_typeCommon = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, dimCommon);
  m_commonModeDatasetIndex = m_writer.createUnlimitedSizeDataset(srcGroup.id(), "common_mode", m_typeCommon, m_dataSetCreationProperties);

  MsgLog(logger,info,"hdfwriterCsPad types: H5Tget_size(uint16[4])= " << H5Tget_size(m_typeArrayDim4uint16)
         << " H5Tget_size(quad)= " << H5Tget_size(m_typeQuad) 
         << " H5Tget_size(element)= " << H5Tget_size(m_typeElement)
         << " H5Tget_size(data)= " << H5Tget_size(m_typeData)
         << " H5Tget_size(common)= " << H5Tget_size(m_typeCommon));
}

void HdfWriter<Psana::CsPad::DataV2>::store(DataTypeLoc dataTypeLoc, 
                                            hdf5pp::Group & srcGroup, 
                                            const PSEvt::EventKey & eventKey, 
                                            PSEvt::Event & evt, 
                                            PSEnv::Env & env) 
{
  MsgLog(logger,fatal,"store not implemented for CsPad::DataV2");
}

void HdfWriter<Psana::CsPad::DataV2>::store_at(DataTypeLoc dataTypeLoc, 
                                               long index, hdf5pp::Group & srcGroup, 
                                               const PSEvt::EventKey & eventKey, 
                                               PSEvt::Event & evt, 
                                               PSEnv::Env & env) 
{
  MsgLog(logger,fatal,"store_at not implemented for CsPad::DataV2");
}

void HdfWriter<Psana::CsPad::DataV2>::append(DataTypeLoc dataTypeLoc,
                                             hdf5pp::Group & srcGroup, const PSEvt::EventKey & eventKey, 
                                             PSEvt::Event & evt, PSEnv::Env & env) 
{
  MsgLog(logger,info,"HdfWriter<Psana::CsPad::DataV2>::append");
  checkType<Psana::CsPad::DataV2>(eventKey, "HdfWriterFrom");

  boost::shared_ptr<Psana::CsPad::DataV2> ptr; 
  if (dataTypeLoc == inEvent) ptr = evt.get(eventKey.src(), eventKey.key()); 
  else if (dataTypeLoc == inConfigStore) ptr = env.configStore().get(eventKey.src());

  const unsigned nquads = ptr->quads_shape()[0];
  unsigned nsect = 0;
  for (unsigned q = 0; q != nquads; ++ q) {
    nsect += ptr->quads(q).data().shape()[0];
  }

  //  const Psana::CsPad::ElementV2 *quadAddress = &(ptr->quads(0));
  //  const void * dataAddress = ptr->quads(0).data().data();
  ndarray<int16_t, 3> data = make_ndarray<int16_t>(nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2);
  unsigned s = 0;
  for (unsigned i = 0; i != nquads; ++ i) {
    const ndarray<const int16_t, 3>& small = ptr->quads(i).data();
    std::copy(small.begin(), small.end(), &data[s][0][0]);
    s += small.shape()[0];
  }

  //  m_writer.append(srcGroup.id(), m_elementDatasetIndex, quadAddress);
  m_writer.append(srcGroup.id(), m_dataDatasetIndex, data.data());
  //  m_writer.append(m_commonModeDatasetIndex, );

}

void HdfWriter<Psana::CsPad::DataV2>::addBlank(hdf5pp::Group & group)
{
  MsgLog(logger,fatal,"addBlank not implemented for CsPad::DataV2");
}
