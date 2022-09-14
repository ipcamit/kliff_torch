cmake_minimum_required(VERSION 3.16)
project(neighbour)

FIND_PACKAGE(EnvModules REQUIRED)
env_module(load python39)

set(CMAKE_CXX_STANDARD 14)

#----Pybind11
find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)
#------------

pybind11_add_module(nl neighbor_list_bind.cpp  neighbor_list.cpp helper.hpp)
