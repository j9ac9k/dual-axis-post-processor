import sys
from cx_Freeze import setup, Executable
import matplotlib

#################
# To build, in cmd or PS run:
# python setup.py build
#################

base = None
if sys.platform == "win32":
    base = "Win32GUI"
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"includes": ["matplotlib.backends.backend_qt4agg",
                                  "scipy.special._ufuncs_cxx", 
                                  "scipy.integrate.vode",
                                  "scipy.integrate.lsoda",
                                  "scipy.sparse.csgraph._validation"],
                     "include_files": [(matplotlib.get_data_path(), "mpl-data")],
                     "excludes": ["pyzmw",
                                  "wxpython",
                                  "tables",
                                  "zmq",
                                  "wx",
                                  "PySide",
                                  "sqlite3",
                                  "Tkinter",
                                  "ipython",
                                  "_ssl",
                                  "PyQt4.QtSvg",
                                  "numpy.core._dotblas"],
                     "optimize": 0,
                     }

executables = [Executable("pyqt4_matplotlib.py", base=base)]

# GUI applications require a different base on Windows (the default is for a
# console application).


setup(name="Post Processing Script",
      version="0.2",
      description="My GUI application!",
      options={"build_exe": build_exe_options},
      executables=[Executable("main.py", base=base)])
