# PASCAl-COCO-convert-tools
conversion tools for CSV2PascalVOC, PascalVOC2COCO

Very useful CSV, pascal and coco format conversion tools to convert non-standard data formats to Pascal and COCO data formats for easy training and testing. Take the remote sensing image dataset NWPU dataset format as an example.

Preprocessing: NWPU2CSV.py converts the original format of the NWPU dataset into CSV format data;
Steps to convert CSV to PascalVOC:
1, CSV2pascalvoc.py, convert the CSV format to PascalVOC format xml tag;
2, CSV2pascalvoc_trval.py, generate a txt file in PascalVOC format;

Steps to convert PascalVOC to COCO:
Run PascalVoc2COCO.py to generate the json file.

非常实用的CSV、pascal和coco格式转换工具，完成不规范数据格式到Pascal和COCO数据格式的转换，便于训练和测试。
以遥感图像数据集NWPU数据集格式为例子。

预处理：NWPU2CSV.py将NWPU数据集原始格式转换为CSV格式数据；
CSV转换为PascalVOC的步骤:
1、CSV2pascalvoc.py，将CSV格式转换为PascalVOC格式xml标签；
2、CSV2pascalvoc_trval.py，生成PascalVOC格式的txt文档；

PascalVOC转换为COCO的步骤:
运行PascalVoc2COCO.py，生成json文件



