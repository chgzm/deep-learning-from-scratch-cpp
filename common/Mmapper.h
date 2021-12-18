#ifndef MMAPPER_H
#define MMAPPER_H

#include <cstdint>
#include <string>

class Mmapper {
public:
    Mmapper();
    ~Mmapper();

    void* mmapReadOnly(const std::string& filePath);

private:
    std::size_t fileSize_;
    void* addr_;  
};

#endif
