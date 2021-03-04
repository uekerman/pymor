#! /bin/bash

this_dir=$(dirname "$0")
export PATH="$this_dir"/usr/bin:"$PATH"
export PKG_CONFIG_PATH="$this_dir"/usr/conda/lib/pkgconfig
export LD_LIBRARY_PATH="$this_dir"/usr/conda/lib

# strace "$this_dir"/usr/bin/python "$this_dir"/script.py "$@"
# "$this_dir"/usr/bin/python -m pdb "$this_dir"/usr/bin/jupyter-qtconsole "$@"
# python "$this_dir"/usr/bin/jupyter-qtconsole "$@"
python "$this_dir"/usr/bin/jupyter-notebook "$@"
