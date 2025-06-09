这是一个使用zluda让pytorch运行再amd显卡的项目，内置rocm5.7 dll文件无需自行安装rocm5.7，并且hook一些不支持再zluda上运行的torch 函数让他跑再cpu上。

使用方法把pytorch-zluda解压到python.exe的目录命名为zluda。

zluda的bin目录应该有zluda和rocm5.7的dll文件和rocm5.7的库，但太大了我打包成压缩包上传使用时解压出来，你可自己手动更换里面的dll文件 这些都是来自rocm 和zluda 和amd驱动的

之后再你的.py文件 直接 """import zluda """ 

注意！！！ import zluda 必须在torch 之前应用不然dll 还是会使用torch 自己的。
