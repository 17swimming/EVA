


核心指标：
frontend_stalls，这个计数正好表示 Core 想取包但 issue_fifo 为空的周期数，每个tile都会有一拍固定的frontend_stalls，因此我们的目标是让frontend_stalls = 分配的tile数——————我已经取消了这一拍的固定开销。


## issue fifo
self.issue_fifo， 存放package，深度为2就足够，深度继续增加不会有性能提升。

## shift_initial_hole_ifmap.py的实现方式
ready_packets -> issue_fifo -> Core
ready_packets在初始化时写入所有package。
后续只要issue_fifo不为空，就pop一个ready_packets到issue_fifo中。

## shift_without_row_order.py的实现方式
使用两个split同时处理两行，split每拍如果生成了package，就写入issue_fifo，issue_fifo依旧还是存package。
而且对于split，可以接收enabled_models，但是依旧只将mode0的结果压入issue fifo，mode1和2只检测，不生成package。

缺点：这个会导致原先的退休策略（只要输入行号增加，就默认最老的那行需要退休）失效了。进一步的，还会导致流式退也失效，因为不能保证PE2中[0:c-1]列是可以退休的了。


## shift.py的实现方式
使用两个split，每个split处理一行,并且每个split都有一个深度为4个package的FIFO，但是严格按照输入行号从这两个split中取package，并把这个过程称为预计算。
对于split来说，如果在预计算时发现了pattern1/2，也正常存入FIFO。
注意：pattern1和pattern2的package一拍就能得到，



## issue
core中的issue逻辑，在发包时，如果是pattern 0，发给PE array（即所有的PU），如果是pattern1/2，就把package发给express unit。

通过test_sim.py验证可知，每个tile都会有一拍固定的frontend_stalls。
