import numpy as np

# 基于两次DC综合的结果，用于快速估算不同配置下面积和功耗开销的
# 现在已经无用

def avg(a, b, c):
    return np.mean([a, b, c])


energy_per_MAC = 0.23  # pJ/MAC ， cite PTB
energy_per_AC = 0.1  # pJ/AC for int32 adder in 45 nm CMOS， cite ISCA 2016 — EIE: Efficient Inference Engine on Compressed Deep Neural Network
# cite yiran chen HPCA19：HyPar: Towards Hybrid Parallelism for Deep Learning Accelerator Array. HPCA 2019
SRAM_access = 5.0 #pJ
DRAM_access = 640 #pJ

# Baseline area/power.
area_of_Prosperity = 6147538
on_chip_power_of_Prosperity = 446.5  # mW，derived from code of prosperity
frequency_of_Prosperity = 500  # MHz

frequency_of_GustavSNN = 1000  # MHz
area_of_GustavSNN = 1340000  # derived from paper
# no power

frequency_of_DLA = 500  # MHz
power_of_DLA = 888


frequency_of_EVA = 500  # MHz


# Config.
oh = 16
ow = 16
psum_width = 16

num_cluster = 2
num_core_per_cluster = 4
num_kernel_per_core = 8

# Power and area.
area_of_splitunit = 16202
power_of_splitunit = 0.68

area_of_issue = 2456
power_of_issue = 0.196

area_of_dispatcher = 2256
power_of_distpatcher = 0.134

BYTE_of_activation_tile_ram = 32768
area_of_activation_tile_ram = 79700
power_of_activation_tile_ram = 0.7     #  偏小
power_of_REG_activation_tile_ram = 105.221

BYTE_of_task_weight_ram = 18324
area_of_task_weight_ram = 169887
power_of_task_weight_ram = 3.838      # 比CACTI的小了10倍
power_of_REG_task_weight_ram = 59.597

# Known register-array point.
now_BYTE = 4992
now_area = 160000       # 算下来后是 4.0006平方微米 per bit
now_power = 1.02        # 算下来后是 

# Psum pool in Spiking Processing Unit.
psum_pool_BYTE_per_core = 6 * ow * psum_width * num_kernel_per_core / 8
# total_psum_pool_BYTE = psum_pool_BYTE_per_core * num_core_per_cluster * num_cluster
psum_pool_area = psum_pool_BYTE_per_core / now_BYTE * now_area
psum_pool_power1 = psum_pool_BYTE_per_core / BYTE_of_task_weight_ram * power_of_REG_task_weight_ram
psum_pool_power2 = psum_pool_BYTE_per_core / BYTE_of_activation_tile_ram * power_of_REG_activation_tile_ram
psum_pool_power3 = psum_pool_BYTE_per_core / now_BYTE * now_power
psum_pool_power = avg(psum_pool_power1, psum_pool_power2, psum_pool_power3)

area_of_Spiking_Processing_Unit = psum_pool_area 
power_of_Spiking_Processing_Unit = psum_pool_power

# core
core_area = area_of_splitunit + area_of_issue + area_of_Spiking_Processing_Unit
core_power = power_of_splitunit + power_of_issue + power_of_Spiking_Processing_Unit
print('core_area:',core_area)
print('core_power',core_power)

# Accumulator.
accumulator_BYTE = num_kernel_per_core * ow * oh  * psum_width / 8
accumulator_area = accumulator_BYTE / now_BYTE * now_area
accumulator_power1 = accumulator_BYTE / BYTE_of_task_weight_ram * power_of_REG_task_weight_ram
accumulator_power2 = accumulator_BYTE / BYTE_of_activation_tile_ram * power_of_REG_activation_tile_ram
accumulator_power3 = accumulator_BYTE / now_BYTE * now_power
accumulator_power = avg(accumulator_power1, accumulator_power2, accumulator_power3)

# cluster
cluster_area = num_core_per_cluster * core_area + accumulator_area
cluster_power = num_core_per_cluster * core_power + accumulator_power

#eva_engine
eva_engine_area = (
    area_of_task_weight_ram
    + area_of_activation_tile_ram
    + area_of_dispatcher
    + num_cluster * cluster_area
)
eva_engine_power = (
    + power_of_task_weight_ram
    + power_of_activation_tile_ram
    + power_of_distpatcher
    + num_cluster * cluster_power
)

print(f"eva_engine_area: {eva_engine_area / 1000000} mm^2")
print(f"eva_engine_power: {eva_engine_power} mW")
