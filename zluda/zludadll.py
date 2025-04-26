
import os
import ctypes


ZLUDA_TARGETS = ('nvcuda.dll',)# 'nvml.dll',"nccl.dll")
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    #'cudart.dll':'cudart64_110.dll',
    #'cusparse.dll':'cusparse64_11.dll',
    #'cufft.dll': 'cufft64_10.dll',
    #'cufftw.dll': 'cufftw64_10.dll',
    #'nvrtc.dll': 'nvrtc64_112_0.dll',
    }


def load_dll() -> None:
    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"
    zluda_path=".\zluda\\bin"

    # 打印 DLL 文件的路径

    for v in ZLUDA_TARGETS:
        ctypes.CDLL(os.path.join(zluda_path, v),mode=ctypes.RTLD_GLOBAL)
        #ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))

    for v in DLL_MAPPING.values():
        #ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
        ctypes.CDLL(os.path.join(zluda_path, v),mode=ctypes.RTLD_GLOBAL)







    