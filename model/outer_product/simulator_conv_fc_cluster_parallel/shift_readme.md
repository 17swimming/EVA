
核心指标：
frontend_stalls，这个计数正好表示 Core 想取包但 issue_fifo 为空的周期，目标是0

## issue fifo
self.issue_fifo， 存放package，深度为2就足够，深度继续增加不会有性能提升。

## shift_initial_hole_ifmap.py的实现方式
ready_packets -> issue_fifo -> Core
ready_packets在初始化时写入所有package。
后续只要issue_fifo不为空，就pop一个ready_packets到issue_fifo中。

## shift.py的实现方式
使用两个split同时处理两行，split每拍如果生成了package，就写入issue_fifo，issue_fifo依旧还是存package。
而且对于split，可以接收enabled_models，但是依旧只将mode0的结果压入issue fifo，mode1和2只检测，不生成package。

缺点：这个会导致原先的退休策略（只要输入行号增加，就默认最老的那行需要退休）失效了。进一步的，还会导致流式退也失效，因为不能保证PE2中[0:c-1]列是可以退休的了。

使用两个split，每个split处理一行。通过test_sim.py验证可知，每个split的处理速度其实不慢，那为什么还会出现这么多frontend_stalls？
