
import os
import ctypes

from pathlib import Path
AMD_DLL=['amdhip64.dll']
HIPSDK_TARGETS = ['rocblas.dll']
ZLUDA_TARGETS = ('nvcuda.dll',)#"nvml.dll","nccl.dll")#"nccl.dll",)# 'nvml.dll')#,"nccl.dll")
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    #'cudnn.dll':'cudnn64_8.dll',
    #'cudart.dll':'cudart64_110.dll',
    #'cusparse.dll':'cusparse64_11.dll',
    #'cufft.dll': 'cufft64_10.dll',
    #'cufftw.dll': 'cufftw64_10.dll',
    #'nvrtc.dll': 'nvrtc64_112_0.dll',
    }


def load_dll(gfx) -> None:

    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"
    zluda_path=os.path.join(Path(__file__).parent, "bin")
    
    if gfx=="gfx803":
        os.environ["ROCBLAS_TENSILE_LIBPATH"] = os.path.join(Path(__file__).parent, "bin","rocblas","library")+"\\"
        dll_580=os.path.join(Path(__file__).parent, "bin","580")
        for v in AMD_DLL:
            ctypes.CDLL(os.path.join(zluda_path, v),mode=ctypes.RTLD_GLOBAL)
        for v in HIPSDK_TARGETS:
            ctypes.CDLL(os.path.join(dll_580, v),mode=ctypes.RTLD_GLOBAL)
        

    for v in ZLUDA_TARGETS:
        ctypes.CDLL(os.path.join(zluda_path, v),mode=ctypes.RTLD_GLOBAL)
    for v in DLL_MAPPING.values():
        ctypes.CDLL(os.path.join(zluda_path, v),mode=ctypes.RTLD_GLOBAL)






    
