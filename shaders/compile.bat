@echo off
REM
REM SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: MIT
REM
REM Permission is hereby granted, free of charge, to any person obtaining a
REM copy of this software and associated documentation files (the "Software"),
REM to deal in the Software without restriction, including without limitation
REM the rights to use, copy, modify, merge, publish, distribute, sublicense,
REM and/or sell copies of the Software, and to permit persons to whom the
REM Software is furnished to do so, subject to the following conditions:
REM
REM The above copyright notice and this permission notice shall be included in
REM all copies or substantial portions of the Software.
REM
REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
REM THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
REM FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
REM DEALINGS IN THE SOFTWARE.
REM
@echo on
glslangValidator.exe -V -S vert --target-env spirv1.6 vert.vert -o vert.spv
glslangValidator.exe -V -S comp --target-env spirv1.6 matvecmul.comp -o matvecmul.spv
glslangValidator.exe -V -S rgen --target-env spirv1.6 matvecmul.comp -DRAYGEN=1 -o matvecmulray.spv
glslangValidator.exe -V -S frag --target-env spirv1.6 matvecmul.comp -DFRAG=1 -o matvecmulfrag.spv
glslangValidator.exe -V -S comp --target-env spirv1.6 matvecmul.comp -DINT8=1 -o matvecmuls8.spv
glslangValidator.exe -V -S rgen --target-env spirv1.6 matvecmul.comp -DINT8=1 -DRAYGEN=1 -o matvecmulrays8.spv
glslangValidator.exe -V -S frag --target-env spirv1.6 matvecmul.comp -DINT8=1 -DFRAG=1 -o matvecmulfrags8.spv
glslangValidator.exe -V -S comp --target-env spirv1.6 matvecmul.comp -DE4M3=1 -o matvecmulfp8.spv
glslangValidator.exe -V -S rgen --target-env spirv1.6 matvecmul.comp -DE4M3=1 -DRAYGEN=1 -o matvecmulrayfp8.spv
glslangValidator.exe -V -S frag --target-env spirv1.6 matvecmul.comp -DE4M3=1 -DFRAG=1 -o matvecmulfragfp8.spv
glslangValidator.exe -V -S comp --target-env spirv1.6 outerproduct.comp -o outerproduct.spv
glslangValidator.exe -V -S rgen --target-env spirv1.6 outerproduct.comp -DRAYGEN=1 -o outerproductray.spv
glslangValidator.exe -V -S frag --target-env spirv1.6 outerproduct.comp -DFRAG=1 -o outerproductfrag.spv
