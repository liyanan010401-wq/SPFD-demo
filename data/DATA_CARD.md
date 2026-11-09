# SYNTHETIC DATA CARD — 合成数据卡

本文件及 SYNTHETIC_sample.npz 中的数据均为人工合成，不是真实侧扫声呐，不来自 SubPipe，不代表任何患者、传感器采集、真实管道或实际物理频率。

生成器复用原 SPFD_demo/data/generate.py 的图像构造：NumPy PCG64、种子1729，24幅24×24单通道图；人工亮矩形、周期背景和高斯噪声组合后裁到[0,1]，最后两行置零。相同seed的图像/框/划分与原例一致；新增synthetic_data、publication_evidence和dataset_id元数据。

字段：
- images：float32 [24,1,24,24]，原始人工强度[0,1]，不是完成z-score后的输入。
- boxes：float32 [24,2,4]，归一化cx,cy,w,h；仅counts指定数量有效。
- counts：每窗0/1/2目标，依样本编号mod3。
- valid：bool [24,1,24,24]，排除末两行，单窗528个有效像素。
- window_id：0–23人工编号；group_id：每两窗一组0–11。
- split：train 12窗/6组，dev 6窗/3组，test 6窗/3组，两两无组交集。
- frequency：train/dev为1、test为2，仅表示两种合成背景纹理，绝不是1/2 kHz或900/455 kHz。
- synthetic_data=True；publication_evidence=False；dataset_id=SYNTHETIC-SPFD-DEMO-v2。

训练只使用train框，供体只来自train同标签异窗。dev在本例中不参与选择，test框仅用于合成框MAE诊断。预处理对每窗有效域min–max，再以train的有效像素均值/总体标准差固定z-score；测试图像不参与统计拟合。invalid在标准化空间归零。

SYNTHETIC_preview.png只是人工输入的预览。它不进入任何论文实测图；矩形边缘不应称为声学阴影或真实管道结构。synthetic_*输出同样不具有论文证据资格。

再生成命令（在包根目录）：
    python -B -X utf8 data/generate.py

正式研究需要另接有合法来源、航迹/复访/双频位置分组和完整窗坐标的真实数据接口；这不是把frequency字段数字改成900/455即可完成的替换。
