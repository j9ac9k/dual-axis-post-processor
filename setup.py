import sys
from cx_Freeze import setup, Executable
import matplotlib

base = None
if sys.platform == "win32":
    base = "Win32GUI"
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"includes": ["matplotlib.backends.backend_qt4agg"],
                     "include_files": [(matplotlib.get_data_path(),
                                        "mpl-data")],
                     "excludes": ["pyzmw",
                                  "wxpython",
                                  "tables",
                                  "zmq",
                                  "wx",
                                  "PySide",
                                  "sqlite3",
                                  "Tkinter",
                                  "ipython",
                                  "scipy.lib.lapack.flapack",
                                  #"numpy.linalg._umath_linalg",
                                  "scipy.lib.blas.fblas",
                                  #"numpy.core._dotblas",
                                  #"numpy.linalg.lapack_lite",
                                  "scipy.sparse._sparsetools",
                                  "_ssl",
                                  #"numpy.core.multiarray"
                                  #"numpy.fft",
                                  #"unicodedata",
                                  #"matplotlib.ft2font",
                                  "PyQt4.QtSvg",
                                  ],
                     "optimize": 0,
                     }

executables = [Executable("pyqt4_matplotlib.py", base=base)]

# GUI applications require a different base on Windows (the default is for a
# console application).


setup(name="Post Processing Script",
      version="0.1",
      description="My GUI application!",
      options={"build_exe": build_exe_options},
      executables=[Executable("main.py", base=base)])
