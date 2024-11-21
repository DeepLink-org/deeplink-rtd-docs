# 硬件测评

<!-- AI底层技术通常分为芯片、计算、框架三个层次，在目前的国际主流AI生态中，英伟达GPU是人工智能计算芯片的领导者，其V100和A100型号的GPU是当今最主流的人工智能计算加速芯片，并基于CUDA生态筑起AI算力的“护城河”。经过多年的政府支持和自主创新，国产软硬件也取得了一定突破，在国内逐渐形成了涵盖计算芯片、开源平台、基础应用、行业应用及产品等环节较完善的人工智能产业链，但是我们仍需正视、重视与英伟达等国际一流企业的技术差距。 -->
<!-- 上海人工智能实验室牵头的 -->
硬件测评基于团体标准评测方法，以国际主流芯片的性能作为对标，对送测芯片进行技术规格、软件生态、功能、性能等多维度测试，并按季度产出硬件评测报告。评测结论可为各类国产加速卡在不同维度的表现提供参考。

### **什么是硬件测评**
<!-- 硬件测评是面向国产深度学习加速卡进行的多维度评测工作。硬件测评提供一套标准的行业测试方法，提供技术规格、软件生态、功能测试、性能测试等多视角，并周期性产出标准测评结果。硬件测评结果可用作各类国产加速卡在不同维度表现的参考。 -->
<!-- 其以《英伟达A100训练测基准测试报告》中相关数据为基准值，着重体现国产训练芯片相比A100基准的差异性(包含优/劣势)。 -->

<!-- 目前硬件测评设计了两个层面的实施方案，分别是[AI芯片评测实施方案](https://deeplink.readthedocs.io/zh-cn/latest/doc/Chip_test/basicmodel.html)和[基于大模型的AI芯片评测实施方案](https://deeplink.readthedocs.io/zh-cn/latest/doc/Chip_test/largecmodel.html)，以应对市场对硬件在不同场景下的能力要求。 -->

硬件测评是面向国产深度学习加速卡进行的多维度评测工作，目前硬件测评设计了两个层面的实施方案，分别是《AI芯片评测实施方案》和《基于大模型的AI芯片评测实施方案》，以应对市场对硬件在不同场景下的能力要求。

#### 特点：
* **全面**：评测广泛涵盖送测芯片的基础技术规格、软件生态、传统模型和算子功能、性能等多方位测试，并专注于深入评估大模型训练、微调、推理支持能力以及模型训练的稳定性指标。
* **先进**：评测方案与人工智能领域的发展同步更新，按季度调整以反映学术和行业进展，并积极吸纳参评单位的合理建议，持续不断地演进和完善。
* **开放**：评测方案和代码完全开放，确保评测过程公平、公正、透明，为所有参与者提供公平竞争和透明的环境。
* **简单**：提供易于执行的评测方案，包括完整的实施指南、数据收集工具和评测代码，并提供及时的技术咨询服务，确保评测流程顺畅无阻。


### **为什么做硬件测评**
硬件测评可作为芯片生产厂商、应用厂商、前场销售及第三方机构对深度学习训练芯片（包含AI芯片模组和AI加速卡等形态）进行设计、采购、评测的参考。

<!-- ### 当前进度和规划

厂商合作进度：目前我们已经和寒武纪、海光、昇腾、燧原、天数、壁仞等硬件厂商达成测评合作。 -->

<!-- 硬件测评工作进度和规划：

![时间线](../../_static/image/Chip_test/CT_milestone.png)
 -->
在整个生态系统中，芯片评测体系作为一个全面、客观的参考指标，不仅是技术发展的推动者，也是产业链条的纽带，促进着技术、市场和用户需求之间的有效连接，助力国内芯片产业的持续繁荣和创新发展。

## **2024Q3评测**

上海人工智能实验室2024Q3芯片评测现已正式启动，诚邀各单位参加！芯片评测工作每季度开展一次，旨在协助芯片厂商更加客观的发现自身软硬件产品的优劣势促进产品迭代，同时为算力市场产品选型提供参考依据。
本季度评测延续了Q2季度的大模型支持能力评测，并基于各企业积极的反馈意见进行了优化更新形成了《基于大模型的AI芯片评测实施方案0.4.1》，新增文生成视频模型相关指标，新增长文本、MoE预训练模型，并调整整体权重设置，具体详见方案。它会与《AI芯片评测实施方案0.6.2》共同构成本次评测的考察范畴。期待通过大模型评测方案的实施，能够为国产硬件的发展注入新的动力，推动大模型技术的普及和应用，促进软硬件协同创新和进步。

### **评测周期**
2024年11月21日开始，2024年12月20日数据提交关闭，2025.1月下旬完成报告发放。

### **评测流程**

<div align="center">
  <img src="../../_static/image/Chip_test/pipeline.png" />
</div>

| 序号 |  流程  |  相关文件  | 截止时间 |
| ---- |  ------  |  ------  |  ----  |
| 1 |  适配参考代码  |  [AIChipBenchmark](https://github.com/DeepLink-org/AIChipBenchmark) |  -  |
| 2 |  基准值log日志  |  [OneDrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/zoutong_pjlab_org_cn/EpBZfyviosVCleMXEUEa7kgBlkp4aioFtU4YkeSIB1MvYw?e=kFKhu1)  |  -  |
| 3 |  基准镜像  |  <li>[Basic Model](https://hub.docker.com/repository/docker/deeplinkaibenchmark/basicmodel/general) </li><li>[Large Model](https://hub.docker.com/repository/docker/deeplinkaibenchmark/llmodel/general)</li>  |  -  |
| 4 |  阅读评测方案，如有问题欢迎沟通。  |  <li>[基于大模型的AI芯片评测实施方案（0.4.2）](https://aicarrier.feishu.cn/wiki/TNhTw7xJfiGfa4kiYuKcZuq9nVe?office_edit=1) </li><li>[AI芯片评测实施方案（0.6.3）](https://aicarrier.feishu.cn/wiki/JWlgwKpjyiaFvVkZ5zTcCf1dnbe?office_edit=1)</li>  |  -  |
| 5 |  反馈本次评测参与情况并完成《预填写表格》填写并提交至指定位置。  |  下载填写：[2024Q4-芯片评测预填写表格.xlsx](https://pjlab-my.sharepoint.cn/:x:/g/personal/hubingying_pjlab_org_cn/ES80c1FF2bFKhL5aRru25gsBjAu0vovuZMtwiXP4EtSNlQ?e=65Snxd)  |  2024.11.25  |
| 6 |  测评开始，基于实施方案对自家芯片进行测试，并进行数据汇总，完成《数据收集表》填写；厂商提交数据汇总表和相关验证材料至指定位置。  | <p>下载填写：</p><li>[2024Q3-大模型数据收集表0.4.2](https://aicarrier.feishu.cn/wiki/Jf0vw9SzmitrIMk3vdjcoJvWnYf?from=from_copylink) </li><li>[2024Q3-基础模型数据收集表0.6.3](https://aicarrier.feishu.cn/wiki/TZjxwAuHpihSVekDC38cey2XnQd?from=from_copylink)</li><li>每个测试项目请务必填写log日志相对路径，方便核验查找</li>  |  2024.12.20  |
| 7 |  实验室进行结果核验和上机复测，完成单芯片评测报告整理，点对点发放。  | -  |  2025.1月下旬  |
| 8 |  厂商可对评测结果和方案合理性进行意见反馈（word形式）  | - |  评测期间  |




### **评测结果和相关材料上传地址**

点对点通知。


<!-- 
1. 季度测评开始前，联系硬件测评工作人员(或邮件联系\"deeplink_benchmark@pjlab.org.cn\")，确认参与本季度测评
2. 季度测评开始，参与测评的芯片请阅读“[测评标准&实施方案](https://aicarrier.feishu.cn/wiki/WOMuwRlF6ilBf5kug8DcbpZwnqb?from=from_copylink)”，基于实施方案对自家芯片进行测试； 
3. 厂商提交数据和验证材料，实验室会进行结果核验； 
4. 实验室完成单芯片评测报告整理（可参考：[报告模版](https://aicarrier.feishu.cn/wiki/R970wOBEhihaoakWkuMco9ognu7)），点对点发放。  -->
<!-- 
### 相关链接
* **测评方案**：[测评标准&实施方案](https://aicarrier.feishu.cn/wiki/WOMuwRlF6ilBf5kug8DcbpZwnqb?from=from_copylink)
* **测评仓库**：[AIChipBenchmark](https://github.com/DeepLink-org/AIChipBenchmark) -->

## 联系我们

如果您所代表的芯片厂商，也期望能够参与硬件测评，可联系社区：deeplink@pjlab.org.cn。
