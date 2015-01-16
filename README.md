## 备注 ##

* 使用GNU profiler查找性能瓶颈
    * 编译时添加-pg
    * ./prog运行，结束时会生成gmon.out
    * gprof ./prog gmon.out | less 查看
* 根据32位或64位主机调整libblas库
    * make blas BITS=[32|64]

* 添加新的内部repo
    * git remote set-url local ssh://git@bioserver:7722/home/git/deep_learning.git
