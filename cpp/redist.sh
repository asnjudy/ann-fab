#!/bin/bash
mkdir -p build
cd build
cmake ..
make
cp lib/_annfab.so ../../annfab
cd ..

