#coding:utf-8

continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)
cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46, 231, 4008, 7393]
cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]

def process(features, feat_dict_):
    featI = []
    featV = []
    # MinMax标准化连续型数据
    for idx in continuous_range_:
        if features[idx] == '':
            featI.append(0)
            featV.append(0.0)
        else:
            featI.append(feat_dict_[idx])
            featV.append((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
    # 类别型特征
    for idx in categorical_range_:
        key = features[idx]
        if key == '' or key not in feat_dict_:
            featI.append(0)
            featV.append(0.0)
        else:
            featI.append(feat_dict_[key])
            featV.append(1.0)
    return (featI, featV)
