这是一个使用zluda让pytorch运行再amd显卡的项目，内置rocm5.7 dll文件无需自行安装rocm5.7，并且hook一些不支持再zluda上运行的torch 函数让他跑再cpu上。

使用方法把pytorch-zluda解压到python.exe的目录命名为zluda。

zluda的bin目录应该有zluda和rocm5.7的dll文件和rocm5.7的库，但太大了请到releases下载完整的，你可自己手动更换里面的dll文件，这些都是来自rocm 和zluda 和amd驱动的

解压到python.exe同一个目录下就行了
-workenv
--python.exe
--zluda
---__init__.py
---zluda.py
---zludadll.py
目录结构大概就是这样

之后在你的.py文件 直接 """import zluda """ 

注意！！！ import zluda 必须在torch 之前应用不然dll 还是会使用torch 自己的。

本项目使用的rocm库和dll和部分代码来自：

https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU

https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU

https://github.com/lshqqytiger/ZLUDA

https://github.com/likelovewant/stable-diffusion-webui-forge-on-amd
