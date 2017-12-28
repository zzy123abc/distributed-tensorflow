# distributed-tensorflow

实验任务:

集群上多节点多GPU分布式训练

CUDA_VISIBLE_DEVICES='' python distributed.py --job_name=ps --task_index=0 

CUDA_VISIBLE_DEVICES='0' python distributed.py --job_name=worker --task_index=0 

CUDA_VISIBLE_DEVICES='1' python distributed.py --job_name=worker --task_index=1 

CUDA_VISIBLE_DEVICES='0' python distributed.py --job_name=worker --task_index=2 

CUDA_VISIBLE_DEVICES='1' python distributed.py --job_name=worker --task_index=3 


实验环境:

Teslak20c集群,使用了3个节点，其中1个节点使用1个cpu作为参数服务器,2个节点分别使用2个gpu作为工作服务器，分布式训练方式可以选择同步和异步两种
