# PASCAl-COCO-convert-tools
conversion tools for CSV2PascalVOC, PascalVOC2COCO

非常实用的CSV、pascal和coco格式转换工具，完成不规范数据格式到Pascal和COCO数据格式的转换，便于训练和测试。
以遥感图像数据集NWPU数据集格式为例子。

预处理：NWPU2CSV.py将NWPU数据集原始格式转换为CSV格式数据；
CSV转换为PascalVOC的步骤:
1、CSV2pascalvoc.py，将CSV格式转换为PascalVOC格式xml标签；
2、CSV2pascalvoc_trval.py，生成PascalVOC格式的txt文档；

PascalVOC转换为COCO的步骤:
1、运行PascalVoc2COCO.py，生成json文件



