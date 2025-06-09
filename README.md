这是一个使用zluda让pytorch运行再amd显卡的项目，并且hook一些不支持再zluda上运行的torch 函数让他跑再cpu上。
使用方法再.py 直接 """import zluda ""
注意！！！ import zluda 必须再torch 之前应用不然dll 还是会使用torch 自己的。
