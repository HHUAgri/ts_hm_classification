# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os, sys
import argparse
import numpy as np
import pandas as pd

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal, ogr, osr

# Enable GDAL/OGR exceptions
gdal.UseExceptions()
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")


def create_field(shp_layer, field_name, field_type):
    """
    创建新的字段
    :param shp_layer:
    :param field_name:
    :param field_type:
    :return:
    """

    # print info
    layer_definition = shp_layer.GetLayerDefn()

    # if exits, delete it first.
    shp_layer.DeleteField(layer_definition.GetFieldIndex(field_name))
    # ds.ExecuteSQL("ALTER TABLE my_shp DROP COLUMN my_field")

    # Add a new field
    new_field = ogr.FieldDefn(field_name, field_type=field_type)
    shp_layer.CreateField(new_field)

    return shp_layer


def assign_field(shp_path, csv_path, type_field, join_field='ID_PARCEL'):
    """
    将CSV文件中的类别信息复制到SHP文件的指定字段
    :param shp_path: 待写入SHP文件路径
    :param csv_path: 分类CSV文件路径
    :param type_field: 写入的字段
    :param join_field: 连接字段
    :return: SHP文件路径
    """
    print(f'### Assigning value for field: %s' % type_field)

    # 1. load shpfile
    shp_dataset = ogr.Open(shp_path, gdal.OF_VECTOR | gdal.OF_UPDATE)
    if shp_dataset is None:
        print(f"### ERROR: could not open {shp_path}")
        return None
    shp_layer = shp_dataset.GetLayer(0)

    # 2. create field
    shp_layer = create_field(shp_layer, type_field, field_type=ogr.OFTInteger)

    # 3. read attributes into dict
    csv_df = pd.read_csv(csv_path, sep=',', header=0)
    type_dict = csv_df.set_index([join_field])[type_field].to_dict()

    # 4. write attributes
    for feat in shp_layer:
        join_value = int(feat.GetField(join_field))
        type_value = type_dict.get(join_value, None)
        if type_value:
            feat.SetField(type_field, int(type_value))
            shp_layer.SetFeature(feat)
        else:
            gt_value = int(feat.GetField('GT_CODE'))
            feat.SetField(type_field, int(gt_value))
            shp_layer.SetFeature(feat)

    # 5. close
    shp_layer = None
    shp_dataset = None
    return shp_path


def main():
    print("##########################################################")
    print("###  #####################################################")
    print("##########################################################")

    shp_path = r'K:\FF\application_dataset\2020-france-agri-hmc\parcel-result\parcel_utm-result.shp'
    csv_path = r'K:\FF\application_dataset\2020-france-agri-hmc\parcel-result\parcel_utm_xgb.csv'
    new_field = 'XGB_TYPE'
    join_field = 'ID_PARCEL'

    # doing
    assign_field(shp_path, csv_path, new_field, join_field)

    # close
    print("### Complete! #############################################")


if __name__ == "__main__":
    main()
