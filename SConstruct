#!/usr/bin/env python
import os
import sys

env = SConscript("godot-cpp/SConstruct")

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

cuda_path = '/usr/local/cuda/'

env.Replace(CC=f"{cuda_path}bin/nvcc -ccbin='g++-13' -arch=native -rdc=true -lstdc++")
#env.Append(LIBS=['curand'])
env.Append(CPPDEFINES=['GDExtensions'])
env.Replace(LINKFLAGS=['-lstdc++'])
env.Append(CPPPATH=["src/", cuda_path + "targets/x86_64-linux/include/"])

sources = Glob("src/*.*[up]")

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "godot_bin/NN.{}.{}.framework/libgdexample.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "godot_bin/NN{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)
