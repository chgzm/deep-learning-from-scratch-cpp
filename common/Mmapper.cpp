#include "Mmapper.h"

#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

Mmapper::Mmapper() 
  : fileSize_(0),
    addr_(nullptr) {
}

Mmapper::~Mmapper() {
    if (addr_ != nullptr && fileSize_ != 0) {
        if (munmap(addr_, fileSize_) != 0) {
            std::fprintf(stderr, "munmap failed.\n");
        }
    }
}

void* Mmapper::mmapReadOnly(const std::string& filePath) {
    const int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "open failed.\n");
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::fprintf(stderr, "fstat failed.\n");
        return nullptr;
    }
    fileSize_ = sb.st_size;

    addr_ = mmap(nullptr, fileSize_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr_ == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed\n");
        return nullptr;
    }

    if (close(fd) == -1) {
        std::fprintf(stderr, "close failed.\n");
        return nullptr;
    }

    return addr_;
}
