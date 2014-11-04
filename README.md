## 备注 ##

* 使用GNU profiler查找性能瓶颈
    * 编译时添加-pg
    * ./prog运行，结束时会生成gmon.out
    * gprof ./prog gmon.out | less 查看
