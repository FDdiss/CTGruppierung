import cupy as cp
import cupyx.scipy.ndimage as cnd
gpu=0 #GPU0, GPU1, GPU2
with cp.cuda.Device(gpu):
    #openingclosing
    print("CUDA Device:", gpu)
    struc=cp.ones((42,42,42))
    print("pos1")
    Array3D = cp.random.rand(1000,1000,1000, dtype="float32")  #1000*1000*1000*4Byte=4Gbyte VRAM
    print("pos2")
    print(Array3D.nbytes/1000000,"MB")
    print("pos3")
    Array3D=cnd.grey_opening(Array3D,footprint=struc) #GPU grey opening

