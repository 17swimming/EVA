"""论文实验共用的蓝粉配色；所有数据系列的颜色在此统一维护。"""

# PALETTE = (
#     "#506AAA",
#     "#728DC1",
#     "#91BBDF",
#     "#B6DBF1",
#     "#F6D3D9",
#     "#E79FBF",
#     "#E17692",
# )
# PALETTE = (
#     "#FFE8E8",
#     "#FF9696",
#     "#FF3030",
#     "#A80000",
#     "#800000",
#     "#AF1681",
#     "#300D26",
#     "#AF5D16",
#     "#FF9900",
# )
PALETTE = (
    "#FDD29C",
    "#8E8184",
    "#D497C3",
    "#F4BCA8",
    "#697381",
    "#CECBCD",
    "#FCEA93",
    "#FEFEFE",
    "#FFD0E9"
)

# 同一加速器在性能、能效和稀疏度实验中使用相同颜色。
ACCELERATOR_COLORS = {
    "EVA": PALETTE[6],
    "NVDLA": PALETTE[4],
    "Prosperity": PALETTE[5],
    "GustavSNN": PALETTE[2],
    "PTB": PALETTE[0],
    # "SpinalFlow": PALETTE[4],
    # "PHI": PALETTE[5],
    "Jetson (original)": PALETTE[8],
    "Jetson (optimized)": PALETTE[1],
}

# 设计空间探索按照配置顺序取色，兼顾深浅与蓝粉对比。
CONFIGURATION_COLORS = tuple(PALETTE[index] for index in (0, 2, 6, 1, 5, 3, 4))
NETWORK_COLORS = {
    "TIM_NCARS": PALETTE[0],
    "QKFormer_DVSGesture": PALETTE[1],
    "SpikeBRGNet_DDD17": PALETTE[2],
    "SDTrack_FE108": PALETTE[5],
    "SpikeET_FE108": PALETTE[5],
    "EMSYOLO_GEN1": PALETTE[6],
}
ABLATION_COLORS = dict(zip(("m0", "m0+m1", "m0_m2", "m0+m1+m2"),
                           (PALETTE[0], PALETTE[5], PALETTE[2], PALETTE[6])))
BREAKDOWN_COLORS = {
    "DRAM": PALETTE[0],
    "scheduler": PALETTE[2],
    "split_unit": PALETTE[3],
    "current_accumulating_sub_unit": PALETTE[6],
    "accumulator": PALETTE[8],
    "LIF": PALETTE[5],
}
