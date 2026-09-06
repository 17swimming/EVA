from collections import deque

import torch


class split:
    """Sparse row encoder with mode0/mode1/mode2 support.

    mode0: normal sliding-window MAC stream.
    mode1: isolated nonzero point.
    mode2: a run of at least four consecutive 1-valued points. The run length is
    encoded in three bits and capped at seven, matching the full1 simulator.
    """

    def __init__(self, kernel_size=3):
        """初始化稀疏行编码器
        
        Args:
            kernel_size (int): 卷积核大小，默认为3
        """
        self.k = kernel_size

    def process(self, if_line, r: int):
        """处理输入行，进行稀疏编码
        
        将输入行编码为三种模式的组合：
        - mode0: 普通滑动窗口MAC流
        - mode1: 孤立的非零点
        - mode2: 连续4个或更多值为1的点的运行长度编码
        
        Args:
            if_line (torch.Tensor or list): 输入特征映射行
            r (int): 当前行索引
            
        Returns:
            tuple: (combined_bitstream, r_array, combined_c, combined_mode)
                - combined_bitstream: 合并后的比特流
                - r_array: 行索引数组
                - combined_c: 列索引数组
                - combined_mode: 模式数组(0/1/2)
        """
        if not isinstance(if_line, torch.Tensor):
            if_line = torch.tensor(if_line, dtype=torch.int8)

        if_line = if_line.view(-1)
        width = len(if_line)
        work_line = if_line.clone()

        mode0_bitstream = []
        mode1_c = []
        mode1_bitstream = []
        mode2_c = []
        mode2_bitstream = []
        c_set = set()
        max_c = width - self.k

        last_idx = -1
        have_nz = False

        while True:
            nz_mask = work_line != 0
            if not nz_mask.any():
                break

            have_nz = True
            curr_idx = nz_mask.nonzero(as_tuple=True)[0][0].item()
            zero_count = curr_idx - last_idx - 1

            run_len = 0
            while curr_idx + run_len < width and work_line[curr_idx + run_len] == 1:
                run_len += 1

            # mode2: 连续4个及以上全1点
            if run_len >= 4:
                process_len = min(run_len, 7)
                mode2_c.append(curr_idx)
                mode2_bitstream.extend(
                    [
                        (process_len >> 2) & 1,
                        (process_len >> 1) & 1,
                        process_len & 1,
                    ]
                )
                work_line[curr_idx : curr_idx + process_len] = 0
                last_idx = curr_idx
                continue

            # mode1: 孤立非零点（前后至少有2个零）
            is_mode1 = (
                zero_count >= 2
                and curr_idx < width - 2
                and work_line[curr_idx + 1] == 0
                and work_line[curr_idx + 2] == 0
            )

            if is_mode1:
                mode1_c.append(curr_idx)
                mode1_bitstream.append(work_line[curr_idx].item())
            else:
                # mode0: 普通滑动窗口模式
                if zero_count >= 2:
                    mode0_bitstream.extend([0, 0])
                else:
                    mode0_bitstream.extend([0] * zero_count)
                mode0_bitstream.append(work_line[curr_idx].item())

                start_k = max(0, curr_idx - (self.k - 1))
                end_k = min(curr_idx, max_c)
                for c in range(start_k, end_k + 1):
                    c_set.add(c)

            work_line[curr_idx] = 0
            last_idx = curr_idx

        # 添加尾部零填充
        if have_nz and mode0_bitstream:
            tail_zeros = width - 1 - last_idx
            if tail_zeros > 0:
                if tail_zeros >= 2:
                    mode0_bitstream.extend([0, 0])
                else:
                    mode0_bitstream.extend([0] * tail_zeros)

        # 合并所有模式的结果
        mode0_c = sorted(c_set)
        combined_c_list = mode0_c + mode1_c + mode2_c
        combined_bitstream_list = mode0_bitstream + mode1_bitstream + mode2_bitstream
        combined_mode_list = [0] * len(mode0_c) + [1] * len(mode1_c) + [2] * len(mode2_c)

        combined_bitstream = torch.tensor(combined_bitstream_list, dtype=torch.int8)
        r_array = torch.tensor([r] * len(combined_c_list), dtype=torch.int16)
        combined_c = torch.tensor(combined_c_list, dtype=torch.int16)
        combined_mode = torch.tensor(combined_mode_list, dtype=torch.int8)

        return combined_bitstream, r_array, combined_c, combined_mode


class SplitUnit:
    """Pipelined split front-end inherited from plus.

    The downstream Core still consumes one encoded row/block at a time through
    row_fifo. Multiple decode slots can prepare rows in parallel so the issue
    stage can stay busy when the encoded row length is long enough.
    """

    def __init__(
        self,
        kernel_size=3,
        w=32,
        fifo_depth=2,
        sram_vec_capacity=10,
        fetch_rows=4,
        decode_lanes=None,
        rob_depth=None,
    ):
        """初始化流水线分割前端
        
        Args:
            kernel_size (int): 卷积核大小，默认为3
            w (int): 特征映射宽度，默认为32
            fifo_depth (int): 行FIFO深度，默认为2
            sram_vec_capacity (int): SRAM向量容量，默认为10
            fetch_rows (int): 预取行数，默认为4
            decode_lanes (int): 解码通道数，默认为fetch_rows
            rob_depth (int): 重排序缓冲区深度，默认自动计算
        """
        self.k = kernel_size
        self.w = w
        self.split = split(kernel_size=kernel_size)
        self.hazard_num = 0  # 冒险计数

        # FIFO和缓冲区配置
        self.row_fifo = deque()
        self.fifo_depth = fifo_depth
        self.sram_vec_capacity = sram_vec_capacity
        self.fetch_rows = fetch_rows
        self.decode_lanes = max(1, int(fetch_rows if decode_lanes is None else decode_lanes))
        self.rob_depth = max(
            1,
            int((fetch_rows + self.decode_lanes + fifo_depth) if rob_depth is None else rob_depth),
        )

        # 内部状态队列
        self.if_reg = deque(maxlen=fetch_rows)      # 输入寄存器队列
        self.decode_slots = []                       # 解码槽位列表
        self.completed_rows = {}                     # 已完成行的字典（按序列号排序）

    def init_stream(self, if_map, mode='conv'):
        """初始化数据流
        
        根据模式初始化特征映射数据流，支持卷积模式和线性模式
        
        Args:
            if_map (torch.Tensor): 输入特征映射
            mode (str): 操作模式，'conv'表示卷积模式，'linear'表示线性模式
        """
        self.if_map = if_map
        self.mode = mode
        self.current_r = 0
        self.row_fifo.clear()
        self.if_reg.clear()
        self.decode_slots.clear()
        self.completed_rows.clear()
        self.is_finished = False

        self.processing_cycles_left = 0
        self.current_row_data = None
        self.next_dispatch_seq = 0
        self.next_emit_seq = 0

        if mode == 'conv':
            # 卷积模式：找出所有非零行
            row_nz_mask = torch.sum(if_map != 0, dim=1) > 0
            self.valid_row_indices = torch.nonzero(row_nz_mask).view(-1).tolist()
            self.valid_row_ptr = 0
        elif mode == 'linear':
            # 线性模式：按SRAM容量分块，找出包含非零元素的块
            self.valid_block_bases = []
            for r in range(0, if_map.shape[0], self.sram_vec_capacity):
                end_idx = min(r + self.sram_vec_capacity, if_map.shape[0])
                if torch.any(if_map[r:end_idx] != 0):
                    self.valid_block_bases.append(r)
            self.valid_block_ptr = 0
        else:
            raise ValueError(f"Unsupported split mode: {mode}")

    def _process_single_row(self, row_data, r):
        """处理单行数据（卷积模式）
        
        对单行输入数据进行稀疏编码处理
        
        Args:
            row_data (torch.Tensor): 行数据
            r (int): 行索引
            
        Returns:
            dict or None: 包含编码结果的字典，若无可编码数据返回None
        """
        bitstream, r_array, c_array, mode = self.split.process(row_data, r)

        if len(r_array) == 0:
            return None

        return {
            'bitstream': bitstream,
            'r': r_array,
            'c': c_array,
            'mode': mode,
        }

    def _process_linear_block(self, tb_block, base_tb, nz_indices):
        """处理线性块（线性模式）
        
        将线性层的输入块转换为编码指令格式
        
        Args:
            tb_block (torch.Tensor): 输入块数据
            base_tb (int): 块的起始索引
            nz_indices (torch.Tensor): 非零元素索引
            
        Returns:
            dict or None: 包含编码结果的字典，若无可编码数据返回None
        """
        insts = {'r': [], 'c': [], 'mode': [], 'bitstream': []}
        base_r = 2  # 线性模式下的起始行偏移

        for idx in nz_indices:
            tb = base_tb + idx.item()
            c = tb % self.w                    # 计算列位置
            r = base_r + (tb // self.w) * 3    # 计算行位置（每个块占3行）

            insts['c'].append(c)
            insts['r'].append(r)
            insts['mode'].append(0)            # 线性模式固定使用mode0
            insts['bitstream'].extend(tb_block[idx].tolist())

        if not insts['r']:
            return None

        return {
            'r': torch.tensor(insts['r'], dtype=torch.int16),
            'c': torch.tensor(insts['c'], dtype=torch.int16),
            'mode': torch.tensor(insts['mode'], dtype=torch.int8),
            'bitstream': torch.tensor(insts['bitstream'], dtype=torch.int8),
        }

    def tick(self):
        """时钟周期推进
        
        根据当前模式调用对应的时钟处理函数
        """
        if self.is_finished:
            return

        if self.mode == 'conv':
            self._tick_conv()
        elif self.mode == 'linear':
            self._tick_linear()

    def _tick_conv(self):
        """卷积模式时钟周期处理
        
        执行预取、分发、解码推进和发射的完整流水线
        """
        if self.is_finished:
            return

        self._prefetch_conv_row()
        self._dispatch_conv_decode_jobs()
        self._advance_decode_slots()
        self._emit_completed_rows()

        # 检查是否完成所有处理
        if (
            self.current_r >= self.if_map.shape[0]
            and len(self.if_reg) == 0
            and len(self.decode_slots) == 0
            and len(self.completed_rows) == 0
        ):
            self.is_finished = True

    def _tick_linear(self):
        """线性模式时钟周期处理
        
        执行预取、分发、解码推进和发射的完整流水线
        """
        if self.is_finished:
            return

        self._prefetch_linear_block()
        self._dispatch_linear_decode_jobs()
        self._advance_decode_slots()
        self._emit_completed_rows()

        # 检查是否完成所有处理
        if (
            self.current_r >= self.if_map.shape[0]
            and len(self.if_reg) == 0
            and len(self.decode_slots) == 0
            and len(self.completed_rows) == 0
        ):
            self.is_finished = True

    def _prefetch_conv_row(self):
        """预取卷积行
        
        从输入特征映射中预取非零行到输入寄存器队列
        """
        if self.valid_row_ptr < len(self.valid_row_indices):
            if len(self.if_reg) < self.fetch_rows:
                target_r = self.valid_row_indices[self.valid_row_ptr]
                row_data = self.if_map[target_r]
                nz_count = int(torch.sum(row_data != 0).item())
                self.if_reg.append((row_data, target_r, nz_count))
                self.valid_row_ptr += 1
                self.current_r = target_r + 1
        else:
            self.current_r = self.if_map.shape[0]

    def _prefetch_linear_block(self):
        """预取线性块
        
        从输入特征映射中预取非零块到输入寄存器队列
        """
        if self.valid_block_ptr < len(self.valid_block_bases):
            if len(self.if_reg) < self.fetch_rows:
                base_r = self.valid_block_bases[self.valid_block_ptr]
                end_idx = min(base_r + self.sram_vec_capacity, self.if_map.shape[0])
                tb_block = self.if_map[base_r:end_idx]

                nz_mask = torch.sum(tb_block != 0, dim=1) > 0
                nz_indices = torch.nonzero(nz_mask).view(-1)

                self.if_reg.append((tb_block, base_r, nz_indices))
                self.valid_block_ptr += 1
                self.current_r = base_r + self.sram_vec_capacity
        else:
            self.current_r = self.if_map.shape[0]

    def _dispatch_conv_decode_jobs(self):
        """分发卷积解码任务
        
        将预取的行数据分发到解码槽位进行处理
        """
        while self._can_dispatch() and len(self.if_reg) > 0:
            row_data, r, nz_count = self.if_reg.popleft()
            row_insts = self._process_single_row(row_data, r)
            self._push_decode_job(row_insts, nz_count)
        self._update_processing_debug_state()

    def _dispatch_linear_decode_jobs(self):
        """分发线性解码任务
        
        将预取的块数据分发到解码槽位进行处理
        """
        while self._can_dispatch() and len(self.if_reg) > 0:
            tb_block, base_tb, nz_indices = self.if_reg.popleft()
            row_insts = self._process_linear_block(tb_block, base_tb, nz_indices)
            self._push_decode_job(row_insts, len(nz_indices))
        self._update_processing_debug_state()

    def _can_dispatch(self):
        """检查是否可以分发新任务
        
        检查解码槽位和ROB深度是否有足够空间
        
        Returns:
            bool: True表示可以分发，False表示不可分发
        """
        in_flight = len(self.decode_slots) + len(self.completed_rows)
        return len(self.decode_slots) < self.decode_lanes and in_flight < self.rob_depth

    def _push_decode_job(self, row_insts, prep_cycles):
        """推送解码任务到解码槽位
        
        Args:
            row_insts (dict): 编码后的行指令数据
            prep_cycles (int): 处理所需的周期数
        """
        job = {
            'seq': self.next_dispatch_seq,      # 任务序列号
            'cycles_left': max(1, int(prep_cycles)),  # 剩余处理周期
            'data': row_insts,                  # 编码数据
        }
        self.next_dispatch_seq += 1
        self.decode_slots.append(job)

    def _advance_decode_slots(self):
        """推进解码槽位
        
        每个解码槽位的剩余周期减1，完成的任务移到completed_rows
        """
        remaining_slots = []
        for job in self.decode_slots:
            job['cycles_left'] -= 1
            if job['cycles_left'] <= 0:
                self.completed_rows[job['seq']] = job['data']
            else:
                remaining_slots.append(job)
        self.decode_slots = remaining_slots
        self._update_processing_debug_state()

    def _emit_completed_rows(self):
        """发射完成的行到FIFO
        
        按序列号顺序将完成的行发射到row_fifo
        """
        # 跳过None结果（空行）
        while self.next_emit_seq in self.completed_rows and self.completed_rows[self.next_emit_seq] is None:
            del self.completed_rows[self.next_emit_seq]
            self.next_emit_seq += 1

        if self.next_emit_seq not in self.completed_rows:
            return

        # 将完成的行添加到FIFO
        if len(self.row_fifo) < self.fifo_depth:
            self.row_fifo.append(self.completed_rows.pop(self.next_emit_seq))
            self.next_emit_seq += 1
        else:
            # FIFO满，产生冒险
            self.hazard_num += 1

    def _update_processing_debug_state(self):
        """更新调试状态信息
        
        更新处理剩余周期和当前行数据的调试信息
        """
        if self.decode_slots:
            self.processing_cycles_left = max(job['cycles_left'] for job in self.decode_slots)
        else:
            self.processing_cycles_left = 0
        self.current_row_data = None
