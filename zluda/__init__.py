import subprocess
import zipfile
import shutil
import os
from pathlib import Path
zluda_version="2025.06.05.11.10"
runing=True
compatibility=True
APU=["gfx902","gfx909","gfx90c","gfx1033","gfx1035","gfx1036","gfx1103"]
AMD=["gfx803","gfx900","gfx90a","gfx902","gfx90c","gfx906","gfx908","gfx940","gfx941","gfx942","gfx1010","gfx1012","gfx1030","gfx1031","gfx1032","gfx1033","gfx1034","gfx1035","gfx1036","gfx1100","gfx1101","gfx1102","gfx1103"]

def get_gfx():
    hipInfo=os.path.join(Path(__file__).parent,"bin","hipInfo.exe")
    lines = subprocess.run(hipInfo, capture_output=True, text=True, encoding="utf-8").stdout.splitlines() 
    gcn_arch_name = None

    for line in lines:
        if "gcnArchName" in line:
            parts = line.split(":")
            if len(parts) > 1:
                gcn_arch_name = parts[1].strip()
                if gcn_arch_name in APU:
                    #第一张卡是核显就继续找第二张
                    continue
                else:
                    #第一张卡不是核显直接返回
                    os.environ["HIP_VISIBLE_DEVICES"] = "0"
                    return gcn_arch_name
    #没有第二张返回核显
    os.environ["HIP_VISIBLE_DEVICES"] = "1"
    return gcn_arch_name
                    
gfx=get_gfx()
def download_lib(gfx):
    workdir = Path(__file__).parent
    zip_path=os.path.join(workdir,"bin","zip",f"{gfx}.zip")
    extract_path=os.path.join(workdir,"bin","rocblas")
    rocm_dir=os.path.join(workdir,"bin","rocblas","library")
    if not os.path.exists(rocm_dir):
        os.makedirs(rocm_dir)
    shutil.rmtree(rocm_dir)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"文件已成功解压到 {extract_path}")
        return True
    except zipfile.BadZipFile:
        print(f"解压失败：{zip_path}")
        return False
    except Exception as e:
        print(f"解压失败：{e}")
        return False

def checklib(gfx):
    rocm_dir=os.path.join(Path(__file__).parent,"bin","rocblas","library")
    for _, _, files in os.walk(rocm_dir):
        for file in files:
            if gfx in file:  
                return True
    return False

def checkenv(gfx):
    if os.path.exists("C:\\Windows\\System32\\nvcuda.dll"):
        if compatibility:
            print("检查到nvcuda.dll使用n卡")
        else:
            raise ValueError("检查到nvcuda.dll使用n卡")        
        #return False
    if not os.path.exists("C:\\Windows\\System32\\amdhip64.dll") and not os.path.exists("C:\\Windows\\System32\\amdhip64_6.dll"):
        if compatibility:
            print("没有发现AMD显卡需要的dll！")
        else:
            raise ValueError("没有发现AMD显卡！")
        return False
    if not gfx in AMD:
        if compatibility:
            print("你的AMD显卡暂时不支持！")
        else:
            raise ValueError("你的AMD显卡暂时不支持！")
        return False
    if checklib(gfx):
        return True    
    else:
        return download_lib(gfx)




if checkenv(gfx):
    import zluda.zludadll as dll
    dll.load_dll(gfx)
    import zluda.zluda as z
    z.initialize_zluda(gfx)
else:
    runing=False
