import subprocess
import zipfile
import shutil
import os



def get_gfx():

    workdir=os.path.join(os.getcwd(), "zluda","bin","hipInfo.exe")
    #print(workdir)
    #output = result.stdout
    # 获取显卡信息
    lines = subprocess.run(workdir, capture_output=True, text=True, encoding="utf-8").stdout.splitlines() 
    gcn_arch_name = None
    for line in lines:
        if "gcnArchName" in line:
            parts = line.split(":")
            #print(line)
            if len(parts) > 1:
                gcn_arch_name = parts[1].strip()
            #print(gcn_arch_name)
            return gcn_arch_name

def download_lib(gfx):
    print("下载")
    workdir = os.getcwd()
    zip_path=os.path.join(workdir, "zluda","bin","zip",f"{gfx}.zip")
    extract_path=os.path.join(workdir, "zluda","bin","rocblas")
    rocm_dir=os.path.join(workdir, "zluda","bin","rocblas","library")
    #判断rocm_dir是否存在
    if not os.path.exists(rocm_dir):
        os.makedirs(rocm_dir)
    #存在就清空内容
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
    rocm_dir=os.path.join(os.getcwd(), "zluda","bin","rocblas","library")

    for _, _, files in os.walk(rocm_dir):
        for file in files:
            if gfx in file:  
                return True
    return False
def checkenv():
    amd=["gfx803","gfx900","gfx90a","gfx902","gfx90c","gfx906","gfx908","gfx940","gfx941","gfx942","gfx1010","gfx1012","gfx1030","gfx1031","gfx1032","gfx1035","gfx1100","gfx1101","gfx1102","gfx1103"]
    #如果amdhip不在就不是amd
    if not os.path.exists("C:\\Windows\\System32\\amdhip64.dll"):
        return False
    #判断显卡型号是否在支持列表
    gfx=str(get_gfx())
    if not gfx in amd:
        return False
    #判断rocm 库文件是否存在


    if checklib(gfx):
        return True    
    else:
        #下载对应的库
        return download_lib(gfx)




if checkenv():
    import zluda.zludadll as dll
    dll.load_dll()
    import zluda.zluda as z
    z.initialize_zluda()
        