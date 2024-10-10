import  numpy as np

class task:
    def __init__(self,
                seq_num,
                size,
                producer,
                priority,
    ):
        self.seq_num=seq_num
        self.size = size
        self.origin_size=size
        self.producer = producer
        self.priority = priority
        self.local = 0
        self.request_num = 0
        self.received_num = 0
        self.flag = 0
class MECS:
    def __init__(self,
                 seq_num,
                 loc_x,
                 loc_y,
                 radius,
                 comp_ability,
                 p_cpu,
    ):
        self.seq_num=seq_num
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.radius = radius
        self.comp_ability = comp_ability
        self.p_cpu = p_cpu
        self.task_queue=[]
        self.percentage_resource=1
        self.w_task=0
        self.eval_score=0

    def  eval_resource(self):
        lenn=0
        if len(self.task_queue) > 0:
            for i in range(len(self.task_queue)):
                lenn=lenn+self.task_queue[i].size
        self.percentage_resourece=(self.p_cpu*1.5*0.001/self.comp_ability-lenn)/(self.p_cpu*1.5*0.001/self.comp_ability)
        if self.percentage_resourece<0:
            return 0
        else:
            return self.percentage_resourece

    def task_process(self):
        if len(self.task_queue)>0:
            i=0
            delete_size = 0
            while i<len(self.task_queue):

                delete_size=delete_size+self.task_queue[i].size
                if delete_size > 0.001 * self.p_cpu / self.comp_ability:
                    break
                i = i + 1
            if i!=len(self.task_queue):

                self.task_queue[i].size=delete_size-0.001*self.p_cpu/self.comp_ability
                del self.task_queue[0:i]
            else:
                self.task_queue.clear()

    def eval_gain(self):
        gain=0
        stand_reward=10
        k=0
        i=0
        if len(self.task_queue)>0:
            p=0
            process_size = 0
            while i<len(self.task_queue):
                process_size=process_size+self.task_queue[i].size
                if process_size >= 0.001 * self.p_cpu / self.comp_ability:
                    break
                i=i+1
            while k<i:
                gain=self.task_queue[k].priority*stand_reward
                k +=1
        return gain

    def eval_w_task(self):
        total_len=0
        if len(self.task_queue) > 0:
            for i in range(len(self.task_queue)):
                total_len=self.task_queue[i].size+total_len
        self.w_task=total_len*self.comp_ability/self.p_cpu


class users:
    def __init__(self,
                 seq_num,
                 loc_x,
                 loc_y,
                 M_rate,
                 M_direction,
                 p_cpu,
                 p_trans,            #transmission power
                 comp_ability,       #Number of cpu cycles required to process each bit of data
                 save_factor,        #Energy Saving Factor
                 parameter,          #Chip Energy Efficiency Parameters
                 W,                  # bandwidths
                 h,                  # Channel Gain
                 β,                 # noise power
                 p_wait,             #Standby power
    ):
        self.seq_num=seq_num
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.next_loc_x = loc_x
        self.next_loc_y = loc_y
        self.M_rate = M_rate
        self.M_direction = M_direction
        self.p_cpu = p_cpu
        self.p_trans = p_trans
        self.comp_ability = comp_ability
        self.save_factor = save_factor
        self.parameter=parameter
        self.W = W
        self.h = h
        self.β = β
        self.p_wait = p_wait
        self.comp_queue=[]
        self.trans_queue=[]
    def create_task(self,
                    num,
                    size,
                    priority,

    ):
        t=task(seq_num=num, size=size, producer=self.seq_num,priority=priority)
        return t

    def decision_offload(self,
                    set_MECS,
                    task,
    ):
        len_comp=0
        len_trans=0
        len_task=0
        serve_set={}
        decision=0
        for k in range(len(set_MECS)):
            distant=np.sqrt(np.square(abs(self.loc_x-set_MECS[k].loc_x))+np.square(abs(self.loc_y - set_MECS[k].loc_y))) #计算距离
            if distant<=set_MECS[k].radius:
                serve_set[set_MECS[k].seq_num]=set_MECS[k]
        #print("距离够了",serve_set)
        for i in range(len(self.comp_queue)):
            len_comp= len_comp+self.comp_queue[i].size

        for j in range(len(self.trans_queue)):
            len_trans=len_trans+self.trans_queue[j].size

        for key,value in list(serve_set.items()):
            for m in range(len(serve_set[key].task_queue)):
                len_task=len_task+serve_set[key].task_queue[m].size
            delay_process=len_task*serve_set[key].comp_ability/serve_set[key].p_cpu
            serve_set[key].w_task=delay_process
            len_task=0

        # offload process
        med_var = 1 + np.square(self.h) * self.p_trans / np.square(self.β)
        rate_trans = self.W * np.math.log(med_var, 2)
        delay_trans = (len_trans + task.size) / rate_trans
        #local process
        delay_local=(len_comp+task.size)*self.comp_ability/self.p_cpu
        power_comput=self.parameter*self.p_cpu*task.size/self.comp_ability
        power_trans=self.p_trans*task.size/rate_trans
        for key,value in list(serve_set.items()):
            power_wait=(value.w_task+task.size*value.comp_ability/value.p_cpu)*self.p_wait
            delay_gain=(delay_local-value.w_task-delay_trans-task.size*value.comp_ability/value.p_cpu)/delay_local
            power_gain=(power_comput-power_wait-power_trans)/power_comput
            total_gain=self.save_factor*delay_gain+(1-self.save_factor)*power_gain
            if  total_gain>0:
                decision=1
                value.eval_score=total_gain
            else:
                del serve_set[key]
        serve_set["decision"]=decision
        return serve_set
    def Move_comp(self):
        if self.M_direction == 0:  # 东
            self.next_loc_x = np.abs(self.loc_x + self.M_rate) % 40
            self.next_loc_y = self.loc_y
        elif self.M_direction == 1:  # 西
            self.next_loc_x = np.abs(self.loc_x - self.M_rate) % 40
            self.next_loc_y = self.loc_y
        elif self.M_direction == 2:  # 南
            self.next_loc_x = self.loc_x
            self.next_loc_y = np.abs(self.loc_y + self.M_rate) % 40
        else:
            self.next_loc_x = self.loc_x
            self.next_loc_y = np.abs(self.loc_y + self.M_rate) % 40
    def Select_MECS(self,
                    overlap_MECS,
                    task,
    ):
        if len(overlap_MECS)==1:
            k=list(overlap_MECS)[0]
            return overlap_MECS[k]
        else:
            self.Move_comp()
        Gain=-1000    #选择服务器收益
        for key,value in list(overlap_MECS.items()):
            current_distance=np.sqrt(np.square(overlap_MECS[key].loc_x-self.loc_x)+np.square(overlap_MECS[key].loc_y - self.loc_y))
            next_distance=np.sqrt(np.square(overlap_MECS[key].loc_x-self.next_loc_x)+np.square(overlap_MECS[key].loc_y - self.next_loc_y))
            Gain_MEC=(current_distance-next_distance)+value.eval_score*self.M_rate
            print("距离差距是",current_distance-next_distance,'\n',"收益是",value.eval_score*self.M_rate)
            if Gain_MEC>Gain:
                Gain=Gain_MEC
                Best_MECS=value
        return Best_MECS

    def task_process(self):
        if len(self.comp_queue)>0:
            i=0
            delete_size = 0
            while i<len(self.comp_queue):

                delete_size = delete_size + self.comp_queue[i].size
                if delete_size > 0.001 * self.p_cpu / self.comp_ability:
                    break
                i = i + 1
            if i!=len(self.comp_queue):
                self.comp_queue[i].size=delete_size-0.001*self.p_cpu/self.comp_ability
                del self.comp_queue[0:i]
            else:
                self.comp_queue.clear()

    def task_trans(self):
        send_queue=[]
        med_var = 1 + np.square(self.h) * self.p_trans / np.square(self.β)
        rate_trans = self.W * np.math.log(med_var, 2)
        if len(self.trans_queue)>0:
            i=0
            trans_size=0
            while i<len(self.trans_queue) :

                trans_size = trans_size + self.trans_queue[i].size
                if trans_size>=0.001*rate_trans:
                    break
                i = i + 1
            if self.trans_queue[0].origin_size != self.trans_queue[0].size and self.trans_queue[0].size < 0.001 * rate_trans:
                self.trans_queue[0].size=self.trans_queue[0].origin_size
            if i!=len(self.trans_queue):
                send_queue=self.trans_queue[0:i]
                self.trans_queue[i].size = trans_size-0.001*rate_trans
                del self.trans_queue[0:i]
            else:
                send_queue=self.trans_queue
                self.trans_queue.clear()
        return send_queue






