def get_area(point):
    length = abs(point[0]-0)
    width = abs(point[1]-0)
    area = length * width
    return area

def get_max_area(points):
    length = np.max(points[:,0])
    width = np.max(points[:,1])
    max_area = length * width
    return max_area

def get_points_num(point,points):
    x_set = points[points[:,0]<point[0]]
    y_set = points[points[:,1]<point[1]]
    intersect_set = [var for var in x_set if var in y_set]
    return len(intersect_set)

def get_weigths(points,alpha,beta):
    max_para = 0
    Rp = len(points)
    Mn = get_max_area(points)
    for point in points:
        Np = get_points_num(point,points)
        Sp = get_area(point)
        if(abs(Np/Rp - Sp/Mn) > max_para):
            max_para = abs(Np/Rp - Sp/Mn)
    weights = (Rp**alpha) * ((1 - max_para)**beta)
    return weights
