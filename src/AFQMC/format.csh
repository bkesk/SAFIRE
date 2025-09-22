#!/bin/csh

module load clang/10.0.1

find . -name '*.h' -print -exec clang-format -i {} \;
find . -name '*.hpp' -print -exec clang-format -i {} \;
find . -name '*.cpp' -print -exec clang-format -i {} \;
find . -name '*.icc' -print -exec clang-format -i {} \;
find . -name '*.cuh' -print -exec clang-format -i {} \;
find . -name '*.cu' -print -exec clang-format -i {} \;

