#!/bin/sh

git submodule update

pip install scons==4.7.0

godot --dump-extension-api
mv ./extension_api.json ./godot-cpp/extension_api.json

echo "run scons platform=<windows|linux|macos> on ./godot-cpp"
echo "then run it on the base folder"
