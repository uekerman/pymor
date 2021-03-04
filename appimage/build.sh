#!/bin/bash
wget -c "https://raw.githubusercontent.com/TheAssassin/linuxdeploy-plugin-conda/master/linuxdeploy-plugin-conda.sh"
wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
chmod u+x linuxdeploy-x86_64.AppImage linuxdeploy-plugin-conda.sh

export ARCH=x86_64
export CONDA_CHANNELS="conda-forge"
export CONDA_DOWNLOAD_DIR=/home/stephan/Repos/pymor/appimage/conda
export CONDA_PACKAGES="pymor;fenics;mshr"

./linuxdeploy-x86_64.AppImage --appdir AppDir -i pymor.png -d pymor.desktop --plugin conda --output appimage --custom-apprun AppRun.sh

cp qt.conf AppDir/usr/bin/
cp kernel.json AppDir/usr/conda/share/jupyter/kernels/python3/

./linuxdeploy-x86_64.AppImage --appdir AppDir  --output appimage
