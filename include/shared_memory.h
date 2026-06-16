#pragma once

#include "hair_types.h"
#include <string>
#include <vector>

namespace hair {

class SharedMemoryBridge {
public:
    SharedMemoryBridge();
    ~SharedMemoryBridge();

    bool create(const std::string& name, uint32_t size);
    bool open(const std::string& name);
    void close();

    bool isOpen() const { return m_isOpen; }
    bool isServer() const { return m_isServer; }
    const std::string& getName() const { return m_name; }
    uint32_t getSize() const { return m_size; }

    bool writeHeader(const SharedMemoryHeader& header);
    bool readHeader(SharedMemoryHeader& header);

    bool writeData(const void* data, uint32_t size, uint32_t offset = 0);
    bool readData(void* data, uint32_t size, uint32_t offset = 0);

    void* getPointer() const { return m_data; }
    uint32_t getDataOffset() const { return sizeof(SharedMemoryHeader); }

    bool waitForFrame(uint32_t timeoutMs = 1000);
    bool signalFrameReady();

private:
    bool mapMemory();
    void unmapMemory();

    std::string m_name;
    uint32_t m_size;
    bool m_isOpen;
    bool m_isServer;

    void* m_handle;
    void* m_data;
};

}
