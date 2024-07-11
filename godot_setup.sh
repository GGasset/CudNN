#!/bin/sh

git submodule update

pip install scons==4.7.0

godot --dump-extension-api
mv ./extension_api.json ./godot-cpp/extension_api.json

{
	cd godot-cpp;
	scons platform=linux custom_api_file=`./extension_api.json` bits=64 -j8
}
