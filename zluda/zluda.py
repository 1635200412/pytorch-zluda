import torch



def angto(input,dt):
    if isinstance(input, torch.Tensor):
        if torch.is_complex(input):
            real_part_bf16 = input.real.to(dtype=dt)
            imag_part_bf16 = input.imag.to(dtype=dt)
            return torch.complex(real_part_bf16, imag_part_bf16)
        else:
            return input.to(dtype=dt)
    return input


_topk = torch.topk
def topk(input: torch.Tensor, *args, **kwargs): 
    device = input.device
    values, indices = _topk(input.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))

_fft_fftn = torch.fft.fftn
def fft_fftn(input: torch.Tensor, *args, **kwargs) -> torch.Tensor: 
    return _fft_fftn(input.cpu(), *args, **kwargs).to(input.device)


_fft_ifftn = torch.fft.ifftn
def fft_ifftn(input: torch.Tensor, *args, **kwargs) -> torch.Tensor: 
    return _fft_ifftn(input.cpu(), *args, **kwargs).to(input.device)


_fft_rfftn = torch.fft.rfftn
def fft_rfftn(input: torch.Tensor, *args, **kwargs) -> torch.Tensor: 
    return _fft_rfftn(input.cpu(), *args, **kwargs).to(input.device)

def jit_script(f, *_, **__):
    f.graph = torch._C.Graph() 
    return f


_torch_sinc= torch.sinc
def torch_sinc(input,*args,**kwargs):
    #print("sinc")
    return _torch_sinc(input.cpu(),*args,**kwargs).to(input.device)

_torch_exp=torch.exp
def torch_exp(input,*args,**kwargs): 
    #print("exp")
    if input.dtype != torch.float32:
        input=angto(input,torch.float32)
        return angto(_torch_exp(input.cpu(),*args,**kwargs).to(input.device),torch.float32)
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return _torch_exp(input.cpu(),*args,**kwargs).to(input.device)


_stft = torch.stft
def stft_fftn(input,*args, **kwargs): # pylint: disable=redefined-builtin
    # print("使用stft")
    if input.dtype != torch.float32:
        input=angto(input,torch.float32)
        kwargs = {k: angto(v.cpu(),torch.float32) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return _stft(input.cpu(),*args,**kwargs).to(input.device)

_istft = torch.istft
def istft_fftn(input, *args, **kwargs,): 
    if input.dtype != torch.float32:
        input=angto(input,torch.float32)
        kwargs["window"]=angto(kwargs["window"].cpu(),torch.float32)
        #kwargs = {k: angto(v.cpu(),torch.float32) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return _istft(input.cpu(),*args,**kwargs).to(input.device)


_fft_irfft= torch.fft.irfft
def fft_irfft(input:torch.Tensor,*args, **kwargs)->torch.Tensor: 
    if input.dtype != torch.float32:
        input=angto(input,torch.float32)
        kwargs = {k: angto(v.cpu(),torch.float32) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return _fft_irfft(input.cpu(),*args, **kwargs).to(input.device)

_fft_rfft=torch.fft.rfft
def fft_rfft(input:torch.Tensor,*args,**kwargs)->torch.Tensor:
    if input.dtype != torch.float32:
        input=angto(input,torch.float32)
        kwargs = {k: angto(v.cpu(),torch.float32) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return _fft_rfft(input.cpu(), *args,**kwargs).to(input.device)
    
_Data_Loader=torch.utils.data.DataLoader
def Data_Loader(*args,**kwargs):
    kwargs["pin_memory"]=False
    return _Data_Loader(*args,**kwargs)




def do_hijack():
    #print("do_hijack")
    torch.version.hip = 5.7

    torch.topk = topk
    torch.jit.script = jit_script
    #torch.fft.fftn = fft_fftn
    #torch.fft.ifftn = fft_ifftn
    #torch.fft.rfftn = fft_rfftn
    torch.stft=stft_fftn
    torch.istft=istft_fftn
    torch.fft.irfft=fft_irfft
    torch.fft.rfft=fft_rfft
    torch.sinc=torch_sinc
    torch.exp=torch_exp
    torch.utils.data.DataLoader=Data_Loader


def de_dnn():
    #print("禁用cudnn")
    do_nothing = lambda _: None
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_flash_sdp = do_nothing
    #math_sdp不能禁用
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_math_sdp = do_nothing

    torch.backends.cudnn.benchmark = False


    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp = do_nothing
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp = do_nothing



def initialize_zluda():



    do_hijack()
    de_dnn()


    

