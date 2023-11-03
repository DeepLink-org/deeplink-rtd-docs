### AI芯片评测实施方案

上海人工智能实验室每季度开展针对国产AI训练芯片的评测工作，多家国产芯片厂商积极参与。实验室对送测芯片进行技术规格、软件生态、功能、性能等多维度测试，并按季度产出硬件评测报告。评测结论可以为算力市场产品选型提供参考依据，同时芯片厂商也可更加客观的发现自身软硬件产品的不足，促进产品迭代。

通过组织AI芯片厂商开展评测工作，可以根本上推进软硬件之间的解耦适配，极大降低算力使用门槛，实现算力要素多样化，为国产芯片高效服务国产大模型保驾护航。 

面向硬件的传统模型评测实施方案的主要评测指标如下：
<table border="2px">
        <tr>
        <th>测试大项</th>
        <th>测试小项</th>
        <th>小项说明</th>
    </tr>
    <tr>
        <td rowspan="4">基本技术规格 </td>
        <td>算力</td>
        <td>考察计算芯片的计算能力，关键指标之一</td>
    </tr>
    <tr>
        <td>内存规格</td>
        <!-- <a href="https://github.com/DeepLink-org/AIChipBenchmark/blob/main/operators/speed_test/communication_bench/readme.md"></a> -->
        <td>考察芯片的显存容量和显存带宽</td>
    </tr>
    <tr>
        <td>通信带宽</td>
        <td>考察芯片的跨卡跨机数据传输能力</td>
    </tr>
    <tr>
        <td>能耗比</td>
        <td>芯片算力与芯片标称功耗的比值</td>
    </tr> 
 	<tr>
        <td rowspan="2">软件生态</td>
        <td>软件栈</td>
        <td>考察芯片对于常用库的支持程度 </td>
    </tr>
    <tr>
        <td>开放性</td>
        <td>考察芯片与业界主流异构计算的模型、接口兼容程度</td>
    </tr>
    <tr>
        <td rowspan="2">功能测试</td>
        <td><a href="https://github.com/DeepLink-org/AIChipBenchmark/tree/main/operators/accuracy_test">算子功能 </a></td>
        <td>考察芯片对算子的支持程度</td>
    </tr>
    <tr>
        <td><a href="https://github.com/DeepLink-org/AIChipBenchmark/blob/main/models/readme.md">模型功能 </a></td>
        <td>考察芯片对基础模型的支持程度</td>
    </tr>
    <tr>
        <td rowspan="3">性能测试</td>
        <td><a href ="https://github.com/DeepLink-org/AIChipBenchmark/blob/main/operators/speed_test/readme.md">算子性能 </a></td>
        <td>考察算子在芯片上的运算时间</td>
    </tr>
    <tr>
        <td>模型性能</td>
        <td>考察模型在芯片上的训练性能</td>
    </tr>
    <tr>
        <td><a href="https://github.com/DeepLink-org/AIChipBenchmark/blob/main/operators/speed_test/communication_bench/readme.md">通信性能</a></td>
        <td>考察算⼦在单节点多芯⽚、多节点多芯⽚条件下的性能表现</td>
    </tr>
</table>