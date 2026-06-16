#include "application.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    hair::Application app;

    int width = 1600;
    int height = 900;
    std::string title = "HairFurSim - GPU Hair Physics Solver";

    if (!app.initialize(width, height, title)) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }

    app.run();
    app.shutdown();

    return 0;
}
