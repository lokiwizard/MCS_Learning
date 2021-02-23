def frange(start,end,interval):
    res = []
    while(start <= end):
        res.append(start)
        start += interval
    return res

class MCS_():
    def __init__(self, position):
        self.speed = 16.7
        self.position = position 
        
        
class EV():
    def __init__(self,remain_elec,need_elec,position):
        self.speed = 16.7 #速度
        self.wait_time = 200 #最长等待时间
        self.max_elec = 90 #最大电量
        self.remain_elec = remain_elec #剩余电量
        self.consume = 0.5 #每经过一个segment的耗电量
        self.position = position #当前位置
        self.need_elec = need_elec #需要的充电量
        
class Search_map(): #计算行驶距离，时间，最优点的类
    def __init__(self, mcs, ev_list, size_x, size_y, x_interval, y_interval): #输入mcs,ev对象，生成搜寻地图的尺寸 ,必须为偶数
        self.mcs = mcs
        self.ev = ev_list
        self.mcs_position = mcs.position
        self.ev_position = [ev.position for ev in ev_list] #所有ev的位置
        self.size_x = size_x
        self.size_y = size_y
        self.x_interval = x_interval
        self.y_interval = y_interval
        self.opt_des = None
        self.points = []
    
    def generate_points(self): #根据mcs的坐标，生成搜寻点集
        min_x = self.mcs_position[0] - self.x_interval*(size_x/2)
        max_x = self.mcs_position[0] + self.x_interval*(size_x/2)
        min_y = self.mcs_position[1] - self.y_interval*(size_y/2)
        max_y = self.mcs_position[1] + self.y_interval*(size_y/2)
        x_points = frange(min_x,max_x,self.x_interval)
        y_points = frange(min_y,max_y,self.y_interval)
        points = [[i,j] for i in x_points for j in y_points] #最终生成的坐标点集
        self.points = np.array(points)
        
    def generate_image(self):
        plt.scatter(x=self.points[:,0],y=self.points[:,1])
        plt.scatter(x=self.mcs_position[0],y=self.mcs_position[1],c='g')
        for i in range(len(self.ev_position)):
            plt.scatter(x=self.ev_position[i][0],y=self.ev_position[i][1],c='r')
        if(self.opt_des is None):
            pass
        else:
            plt.scatter(x=self.opt_des[0],y=self.opt_des[1],c='black')
        plt.grid()
    
    def measure_consume(self, start, des): #获取到达目的地所需的里程和时间,能耗,传入起点和终点
        x_ori = abs(des[0] - start[0]) #x方向移动的距离
        y_ori = abs(des[1] - start[1]) #y方向移动的距离
        segment_count = x_ori/self.x_interval + y_ori/self.y_interval
        segment_size = 400
        time = (segment_count * segment_size)/ self.mcs.speed #单位为秒
        elec_consume = segment_count *self.ev[0].consume #消耗的电量，一个segment消耗0.5kwh
        expense = 2.4 * elec_consume #消耗的电费
        return time,elec_consume,expense #返回耗时和耗电量以及费用
    
    def get_optimal_point(self): 
        optimal_point = None
        extra_expense = 99999999 #初始的费用
        time = 0
        for point in self.points:
            mcs_time,_,mcs_expense = self.measure_consume(self.mcs.position,point)
            flag = 1
            total_expense = mcs_expense #mcs去向目标点需要的费用
            total_time = 0
            for ev in self.ev:
                if(flag):
                    ev_time,ev_elec_consume,ev_elec_consume = self.measure_consume(ev.position,point) #获取每台ev的行驶时间和能耗
                    total_expense += ev_elec_consume  #加上ev行动中损失的电量
                    if(mcs_time > ev_time): #若mcs比ev晚到达，需要考虑ev的等待时间
                        wait_time = mcs_time - ev_time #计算等待时间
                    else:
                        wait_time = 0
                    total_time += wait_time
                    if(wait_time>ev.wait_time or ev_elec_consume>ev.remain_elec):#若等待超时或者耗电超过限额
                        flag = 0
                        break
            if(flag == 1): #若找到符合约束的点
                if(total_expense < extra_expense):
                    extra_expense = total_expense
                    optimal_point = point
                    time = total_time/len(self.ev)
            self.opt_des = optimal_point
        return optimal_point,extra_expense,time
    
size_x = 15
size_y = 15
x_interval = 0.0036
y_interval = 0.0046
