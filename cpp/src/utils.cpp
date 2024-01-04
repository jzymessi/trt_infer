#include <string>
#include <fstream>
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <dirent.h>

struct ImageInfo {
    std::string path;
    std::string id;
};



void findImagesFiles(const std::string& directory, std::vector<std::string>& imageFiles) {
     for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (std::filesystem::is_regular_file(entry.path())) {
            imageFiles.push_back(entry.path().string());
        }
    }
}



