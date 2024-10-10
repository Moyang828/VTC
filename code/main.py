from def_class import  users
from def_class import  MECS
import numpy as np
import  tensorflow as tf
from network import  DoubleDQN
import matplotlib.pyplot as plt

t_slot=0.001

M_rate = 10

tf.compat.v1.disable_eager_execution()

MEMORY_SIZE = 3000


ACTION_SPACE = 1024


sess = tf.compat.v1.Session()

with tf.compat.v1.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=21, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True,learning_rate=0.005,batch_size=32)


sess.run(tf.compat.v1.global_variables_initializer())

# region Init users
data1=np.random.random()
users_A=users(seq_num=0,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.1E+9,
              p_trans=0.1,comp_ability=1/64,save_factor=np.round(data1,1),parameter=34E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=26)

data2=np.random.random()
users_B=users(seq_num=1,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.2E+9,
              p_trans=0.08,comp_ability=1/64,save_factor=np.round(data2,1),parameter=33E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=24)

data3=np.random.random()
users_C=users(seq_num=2,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.15E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data3,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=22)

data4=np.random.random()
users_D=users(seq_num=3,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.12E+9,
              p_trans=0.11,comp_ability=1/64,save_factor=np.round(data4,1),parameter=34E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=21)

data5=np.random.random()
users_E=users(seq_num=4,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data5,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)

data6=np.random.random()
users_F=users(seq_num=5,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data6,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)

data7=np.random.random()
users_G=users(seq_num=6,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data7,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)

data8=np.random.random()
users_H=users(seq_num=7,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data8,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)

data9=np.random.random()
users_I=users(seq_num=8,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data9,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)

data10=np.random.random()
users_J=users(seq_num=9,loc_x=np.random.randint(0,40),
              loc_y=np.random.randint(0,40),M_rate=M_rate,
              M_direction=np.random.randint(0,3),p_cpu=0.14E+9,
              p_trans=0.12,comp_ability=1/64,save_factor=np.round(data10,1),parameter=35E-18,
              W=0.5E+9,h=3,β=1E-3,p_wait=20)




user_set=[users_A,users_B,users_C,users_D,users_E,users_F,users_G,users_H,users_I,users_J]

# region Init MECS

MECS_A=MECS(seq_num=0,loc_x=0,loc_y=10,radius=15,comp_ability=1/64,p_cpu=8E+9)

MECS_B=MECS(seq_num=1,loc_x=12,loc_y=8,radius=12,comp_ability=1/64,p_cpu=6E+9)

MECS_C=MECS(seq_num=2,loc_x=20,loc_y=15,radius=12,comp_ability=1/64,p_cpu=8E+9)

MECS_D=MECS(seq_num=3,loc_x=30,loc_y=25,radius=15,comp_ability=1/64,p_cpu=6E+9)

MECS_E=MECS(seq_num=4,loc_x=35,loc_y=30,radius=16,comp_ability=1/64,p_cpu=8E+9)


MECS_set=[MECS_A,MECS_B,MECS_C,MECS_D,MECS_E]

reward_list_n = []
reward_list_n.append(0)
cost_average = 0
cost_num = 0
def train(RL):
    request_set = [0,0,0,0,0,0,0,0,0,0]
    total_steps=[0,0,0,0,0]
    time = 0
    reward_history = [0,0,0,0,0]
    action_history = [0,0,0,0,0]
    observation_history = np.zeros((5, 21))
    expect_reward = [0,0,0,0,0]
    current_reward = [0,0,0,0,0]
    reward = [0,0,0,0,0]
    action = [0,0,0,0,0]
    observation = np.zeros((5, 21))
    while True:
        for members in MECS_set:
            members.task_process()
            members.eval_w_task()
        for i in range(len(user_set)):
            send_queue = user_set[i].task_trans()
            for item in send_queue:
                if item.received_num==0:
                    MECS_A.task_queue.append(item)
                else:
                    MECS_B.task_queue.append(item)
            user_set[i].task_process()
            if  all(request_set)!=0:

                if request_set[i].flag!=1:
                    task_new = user_set[i].create_task(time, np.random.randint(8,15)*1000000, np.random.randint(1, 3))
                else:
                    task_new = request_set[i]
                    if task_new.priority<6:
                        task_new.priority=task_new.priority+1
            else:
                task_new = user_set[i].create_task(time, np.random.randint(8,15)*1000000, np.random.randint(1, 3))
            serve_set = user_set[i].decision_offload(set_MECS=MECS_set, task=task_new)
            time=time+1
            if serve_set["decision"] == 1:
                task_new.local = 1
                del serve_set["decision"]
                Best_choice = user_set[i].Select_MECS(serve_set, task_new)
                task_new.request_num = Best_choice.seq_num
            else:
                task_new.local = 0
                task_new.request_num = user_set[i].seq_num
                user_set[i].comp_queue.append(task_new)
            request_set[i] = task_new
            user_set[i].Move_comp()
            user_set[i].loc_x = user_set[i].next_loc_x
            user_set[i].loc_y = user_set[i].next_loc_y
            user_set[i].M_direction=np.random.randint(0,3)
        request_MECS = [[0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0
                       ],
                        [0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0
                         ],
                        [0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0
                         ],
                        [0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0
                        ],
                        [0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0, 0, 0,0, 0,0, 0
                         ]
                       ]
        for r in range(len(request_set)):
            if request_set[r].local == 1:
                request_MECS[request_set[r].request_num][2*r]=request_set[r].size/(8*1024*1024)
                request_MECS[request_set[r].request_num][2*r+1]=request_set[r].priority
        for po in range(len(MECS_set)):
            if any(request_MECS[po]) != 0:
                expect_reward[po] = 0
                request_MECS[po].append(MECS_set[po].eval_resource())
                observation[po] = np.array(request_MECS[po])
                action[po] = RL.choose_action(observation[po])
                for q in range(len(request_set)):
                    if request_set[q].local == 1 and request_set[q].request_num == po:
                        m = ''
                        a=action[po]
                        while a > 0:
                            m += str(a % 2)
                            a = a // 2
                        if len(m) < 10:
                            m = m + '0' * (10 - len(m))
                        m = m[::-1]
                        if m[q] == '1':
                            request_set[q].flag = 0
                            request_set[q].received_num = po
                            user_set[q].trans_queue.append(request_set[q])
                            if MECS_set[po].w_task != 0:
                                num = MECS_set[po].w_task/t_slot
                                expect_reward[po]=expect_reward[po]+request_set[q].priority*10/num
                            else:
                                expect_reward[po] = expect_reward[po] + request_set[q].priority * 10
                current_reward[po] = MECS_set[po].eval_gain()
                reward[po] = current_reward[po]+expect_reward[po]
                if total_steps[po] % 2 != 0:
                    RL.store_transition(observation_history[po], action_history[po], reward_history[po], observation[po])
                reward_history[po] = reward[po]+expect_reward[po]
                observation_history[po] = observation[po]
                action_history[po] = action[po]
                total_steps[po]+=1

        if (sum(total_steps)+5)/2 > MEMORY_SIZE:   # learning
            RL.learn()

        if (sum(total_steps)+5)/2 - MEMORY_SIZE > 40000:   # stop game
            break
    return RL.cost_his
q_double = train(double_DQN)









