#点与点的直线距离
def geoDist(pA, pB):
    radLat1 = Rad(pA[0])
    radLat2 = Rad(pB[0])
    delta_lon = Rad(pB[1] - pA[1])
    top_1 = cos(radLat2) * sin(delta_lon)
    top_2 = cos(radLat1) * sin(radLat2) - sin(radLat1) * cos(radLat2) * cos(delta_lon)
    top = sqrt(top_1 * top_1 + top_2 * top_2)
    bottom = sin(radLat1) * sin(radLat2) + cos(radLat1) * cos(radLat2) * cos(delta_lon)
    delta_sigma = atan2(top, bottom)
    distance = delta_sigma * 6378137.0
    return distance
def Rad(d):
    return d * pi / 180.0

#点到直线的距离
def distToSegment(pA, pB, pX):
    d = 0
    a = abs(geoDist(pA, pB))
    b = abs(geoDist(pA, pX))
    c = abs(geoDist(pB, pX))
    p = (a + b + c) / 2.0
    s = sqrt(abs(p * (p - a) * (p - b) * (p - c)))
    if(a!=0):
        d = s * 2.0 / a
    return d

def SlideWindow(enpInit, enpArrayFilter, start, end, cur, m, DMax, count):
    if (end < count):
        d_cur = distToSegment(enpInit[start], enpInit[end], enpInit[cur])  # 当前点到对应线段的距离
        d_m = distToSegment(enpInit[start], enpInit[end], enpInit[m])  # 当前点到对应线段的距离
        if (d_cur > DMax or d_m > DMax):
            enpArrayFilter.append(enpInit[cur])  # 将当前点加入到过滤数组中
            start = cur
            cur = start + 1
            end = start + 2
            m = cur
            d_m = 0
            SlideWindow(enpInit, enpArrayFilter, start, end, cur, m, DMax, count)
        elif ((d_cur <= DMax) and (d_m <= DMax)):
            if (d_cur > d_m):
                m = cur
            cur = end
            end = end + 1
            SlideWindow(enpInit, enpArrayFilter, start, end, cur, m, DMax, count)
            
def trajectory_compress(data):
    max_length = 3800
    dataset = tf.data.Dataset.from_tensor_slices(data)
    data_ = list(dataset.batch(max_length).as_numpy_iterator())
    res = []
    for i in data_:
        i = i.tolist()
        enpArrayFilter = []#过滤数组
        enpArrayFilter.append(i[0])#将第一个点加入过滤数组 
        start = 0
        end = 2
        cur = 1
        m = 1
        count = len(i)-1
        Dmax = 5
        SlideWindow(i,enpArrayFilter,start,end,cur,m,Dmax,count)
        res.extend(enpArrayFilter)
    return res

def reshape_y_hat(y_hat, dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i+j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y

def data_process(signal, file_name):
    if signal == 'geolife':
        f = open(file_name)
        df = pd.read_table(f,skiprows=6,sep=',')
        temp = np.array(df.columns)
        temp = pd.DataFrame([temp],columns=['latitude','longtitude','zero','altitude','time_stamp','date','time'])
        temp[['latitude','longtitude']] = temp[['latitude','longtitude']].astype(np.float64)
        df.columns = temp.columns
        res = pd.concat([temp,df],axis=0,ignore_index=True)
        res = res[['latitude','longtitude']]
        res = res[::]
        return res
    elif signal == 'taxi':
        data = pd.read_table(file_name,header=None,sep=',')
        data.columns = ['id','time','longtitude','latitude']
        #temp = data['time'].str.split(' ')
        #data['date'] = temp.str[0]
        #data['time'] = temp.str[1]
        #date_list = list(set(list(data['date'])))#提取日期
        id = str(data['id'][0])#提取id
        #按日期划分数据
        #data_set = []
        #for date in date_list:
        #    data_ = data[data['date']==date]
        #    data_set.append(data_[['latitude','longtitude']])
        #return {id:data_set}#返回的是按天划分的数据块
        return data[['latitude','longtitude']]


# 计算两个经纬度之间的直线距离
def hav(theta):
    s = sin(theta / 2)
    return s * s
def get_distance_hav(lat0, lng0, lat1, lng1):
    # "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)
 
    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance

def generate_data(data, window_size, step=1):
    res,normalize = NormalizeMult(data, True)
    train_X,train_Y = data_pipeline(res,window_size,step)
    return train_X,train_Y,normalize

#每隔两个坐标点就加入噪声
def random_walk(data):
    flag = 1 #flag为1时，latitude加入噪声，flag为0时，longtitude加入噪声
    for i in range(len(data)):
        if(i%2 == 0):
            if(flag == 0):
                data[i][0] += 0.0036*10
                flag = 1
                continue
            if(flag == 1):
                data[i][1] += 0.0046*10
                flag = 2
                continue
            if(flag == 2):
                data[i][0] += 0.0036*5
                data[i][1] += 0.0046*5
                flag = 0
    return data
