#ifndef TRANSLATOR_EPICS_WRITE_BUFFER_H
#define TRANSLATOR_EPICS_WRITE_BUFFER_H

#include "MsgLogger/MsgLogger.h"
#include "Translator/epics.ddl.h"

namespace EpicsWriteBufferDetails {

template <class T>
  void copyValueFld(T &dest, const T src) { dest = src; }

void copyValueFld(Translator::Unroll::EpicsPvCtrlString::valueBaseType &dest, const char * src);

template <class U>
int getNumberOfStringsForCtrlEnum(const typename U::PsanaSrc &psanaSrc) {
  return -1;
}

template <>
  int getNumberOfStringsForCtrlEnum<Translator::Unroll::EpicsPvCtrlEnum>(const Psana::Epics::EpicsPvCtrlEnum &psanaSrc);

hid_t epicsMemH5Type(int16_t dbrType, int numElements, int numStrsCtrlEnum=-1);
hid_t epicsFileH5Type(int16_t dbrType, int numElements, int numStrsCtrlEnum=-1);

};  // namespace EpicsWriteBufferDetails

namespace Translator {

template <class U>
class EpicsWriteBuffer {
public:
  EpicsWriteBuffer(int16_t dbrType, typename U::PsanaSrc &psanaSrc) 
  {
    m_numElements = psanaSrc.numElements();
    m_dbrType = dbrType;
    m_noStrsForCtrlEnum = EpicsWriteBufferDetails::getNumberOfStringsForCtrlEnum<U>(psanaSrc);
    // this will hold one more value than neccessary since U ends with one value
    m_data = new uint8_t[sizeof(U) + sizeof(typename U::valueBaseType)*m_numElements];
    copyFromPsana(psanaSrc);
  }

  ~EpicsWriteBuffer() { delete m_data; }

  void * data() { return static_cast<void *>(m_data); }

  hid_t getFileH5Type() { return EpicsWriteBufferDetails::epicsFileH5Type(m_dbrType, m_numElements, m_noStrsForCtrlEnum); }
  hid_t getMemH5Type() { return EpicsWriteBufferDetails::epicsMemH5Type(m_dbrType, m_numElements, m_noStrsForCtrlEnum); }

protected:
  void copyFromPsana(typename U::PsanaSrc &src) {
    U *unrollPtr = static_cast<U *>(data());
    copyToUnrolledExceptForValue(src,*unrollPtr);
    int numElementsToCopy = m_numElements;
    if (src.numElements() != m_numElements) {
      numElementsToCopy = std::min(numElementsToCopy, int(src.numElements()));
      MsgLog("EpicsWriteBuffer", error, "epics pv has " 
             << src.numElements() << " which is NOT EQUAL TO numElements at construction: "
             << m_numElements << " using min of two");
    }
    typename U::valueBaseType *valuePtr = &(unrollPtr->value);
    for (int elem=0; elem < numElementsToCopy; ++elem) {
      EpicsWriteBufferDetails::copyValueFld(*valuePtr,src.value(elem));
      valuePtr++;
    }
  }

private:
  int16_t m_dbrType;
  int m_numElements;
  int m_noStrsForCtrlEnum;
  uint8_t *m_data;
};

} // namespace Translator
    
#endif
