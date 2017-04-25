import os
import sys
import zipimport

sys.frozen = True

FILE_NAME = sys.executable
DIR_NAME = os.path.dirname(sys.executable)

TF_CPU_LIB = "%s/lib/python3.5-cpu" % os.path.dirname(DIR_NAME)
TF_GPU_LIB = "%s/lib/python3.5-gpu" % os.path.dirname(DIR_NAME)
if os.path.exists("/usr/local/cuda"):
    tf_lib = TF_GPU_LIB
else:
    tf_lib = TF_CPU_LIB
os.environ["TF_LIBRARY_DIR"] = tf_lib

# Setting the below to '1' or above filters out info. This messages like:
# I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:865] OS X does not support NUMA - returning NUMA node zero
# Setting the below to '2' or above filters out warnings. This hides messages like:
# W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path = [
    "%s/tensorflow/python" % tf_lib,
    "%s/tensorflow/contrib/tfprof/python/tools/tfprof" % tf_lib,
] + sys.path

m = __import__("__main__")
importer = zipimport.zipimporter(os.path.dirname(os.__file__))
name, ext = os.path.splitext(os.path.basename(os.path.normcase(FILE_NAME)))
moduleName = "%s__main__" % name
code = importer.get_code(moduleName)
exec(code, m.__dict__)

versionInfo = sys.version_info[:3]
if versionInfo >= (2, 5, 0) and versionInfo <= (2, 6, 4):
    module = sys.modules.get("threading")
    if module is not None:
        module._shutdown()
