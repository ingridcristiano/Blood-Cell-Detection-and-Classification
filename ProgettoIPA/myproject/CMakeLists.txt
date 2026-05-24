cmake_minimum_required(VERSION 3.10)
project(ImageProcessingProject)

# Imposta lo standard C++17 per abilitare <filesystem>
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CONFIGURAZIONE OPENCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# ------------------------------

# Ora che è dentro la cartella corrente, basta il percorso relativo diretto!
add_subdirectory(3rdparty/ipalib)
add_subdirectory(3rdparty/ucaslib)

# Diciamo al compilatore dove trovare gli header (.h)
include_directories(3rdparty/ipalib)
include_directories(3rdparty/ucaslib)

# Trova tutti i file sorgente (.h, .hpp, .cpp) nella cartella corrente
file(GLOB src *.h *.hpp *.cpp)

# Fix va_start error in VS 14+
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# Crea l'eseguibile principale
add_executable(ipaproject ${src})

# Linka l'eseguibile alle librerie interne e a OpenCV
target_link_libraries(ipaproject ipalib ucaslib ${OpenCV_LIBS})