import numpy as np

def max(a, b):
    if a >= b:
        return a
    else:
        return b

def min(a, b):
    if a <= b:
        return a
    else:
        return b

def cpu_nms(dets, thresh, score_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1 + 1)*(y2 - y1 + 1)
    order = np.argsort(-scores)

    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype = int)

    keep = []
    for ind in range(ndets):
        i = order[ind]
        if suppressed[i] == 1:
            continue
        if scores[i] < score_thresh:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(ind + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep
        
