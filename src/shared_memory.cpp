#include "shared_memory.h"
#include <windows.h>
#include <cstring>
#include <iostream>

namespace hair {

constexpr uint32_t SHARED_MEMORY_MAGIC = 0x48414952;
constexpr uint32_t SHARED_MEMORY_VERSION = 1;

SharedMemoryBridge::SharedMemoryBridge()
    : m_size(0)
    , m_isOpen(false)
    , m_isServer(false)
    , m_handle(nullptr)
    , m_data(nullptr)
{
}

SharedMemoryBridge::~SharedMemoryBridge() {
    close();
}

bool SharedMemoryBridge::create(const std::string& name, uint32_t size) {
    if (m_isOpen) {
        close();
    }

    m_name = name;
    m_size = size;

    m_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        size,
        name.c_str()
    );

    if (m_handle == NULL) {
        std::cerr << "Failed to create shared memory: " << GetLastError() << std::endl;
        return false;
    }

    if (!mapMemory()) {
        CloseHandle(m_handle);
        m_handle = NULL;
        return false;
    }

    m_isServer = true;
    m_isOpen = true;

    SharedMemoryHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = SHARED_MEMORY_MAGIC;
    header.version = SHARED_MEMORY_VERSION;
    header.dataOffset = sizeof(SharedMemoryHeader);
    header.dataSize = size - sizeof(SharedMemoryHeader);

    writeHeader(header);

    return true;
}

bool SharedMemoryBridge::open(const std::string& name) {
    if (m_isOpen) {
        close();
    }

    m_name = name;

    m_handle = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        name.c_str()
    );

    if (m_handle == NULL) {
        std::cerr << "Failed to open shared memory: " << GetLastError() << std::endl;
        return false;
    }

    if (!mapMemory()) {
        CloseHandle(m_handle);
        m_handle = NULL;
        return false;
    }

    m_isServer = false;
    m_isOpen = true;

    SharedMemoryHeader header;
    readHeader(header);

    if (header.magic != SHARED_MEMORY_MAGIC) {
        std::cerr << "Invalid shared memory magic number" << std::endl;
        close();
        return false;
    }

    m_size = header.dataOffset + header.dataSize;

    return true;
}

void SharedMemoryBridge::close() {
    unmapMemory();

    if (m_handle) {
        CloseHandle(m_handle);
        m_handle = NULL;
    }

    m_isOpen = false;
    m_isServer = false;
}

bool SharedMemoryBridge::mapMemory() {
    if (!m_handle) return false;

    m_data = MapViewOfFile(
        m_handle,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        0
    );

    if (m_data == NULL) {
        std::cerr << "Failed to map shared memory view: " << GetLastError() << std::endl;
        return false;
    }

    return true;
}

void SharedMemoryBridge::unmapMemory() {
    if (m_data) {
        UnmapViewOfFile(m_data);
        m_data = nullptr;
    }
}

bool SharedMemoryBridge::writeHeader(const SharedMemoryHeader& header) {
    if (!m_data || !m_isOpen) return false;

    memcpy(m_data, &header, sizeof(SharedMemoryHeader));
    return true;
}

bool SharedMemoryBridge::readHeader(SharedMemoryHeader& header) {
    if (!m_data || !m_isOpen) return false;

    memcpy(&header, m_data, sizeof(SharedMemoryHeader));
    return true;
}

bool SharedMemoryBridge::writeData(const void* data, uint32_t size, uint32_t offset) {
    if (!m_data || !m_isOpen) return false;

    uint32_t dataOffset = sizeof(SharedMemoryHeader) + offset;
    if (dataOffset + size > m_size) return false;

    memcpy((char*)m_data + dataOffset, data, size);
    return true;
}

bool SharedMemoryBridge::readData(void* data, uint32_t size, uint32_t offset) {
    if (!m_data || !m_isOpen) return false;

    uint32_t dataOffset = sizeof(SharedMemoryHeader) + offset;
    if (dataOffset + size > m_size) return false;

    memcpy(data, (char*)m_data + dataOffset, size);
    return true;
}

bool SharedMemoryBridge::waitForFrame(uint32_t timeoutMs) {
    if (!m_isOpen) return false;

    SharedMemoryHeader header;
    readHeader(header);

    uint32_t startFrame = header.frameNumber;
    uint32_t elapsed = 0;

    while (elapsed < timeoutMs) {
        readHeader(header);
        if (header.frameNumber != startFrame) {
            return true;
        }
        Sleep(1);
        elapsed++;
    }

    return false;
}

bool SharedMemoryBridge::signalFrameReady() {
    if (!m_isOpen || !m_isServer) return false;

    SharedMemoryHeader header;
    readHeader(header);
    header.frameNumber++;
    writeHeader(header);

    return true;
}

}
