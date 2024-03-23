import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.wrappers import TransformReward

num=0


def rewardChange(r,a,b):
  if (r==0):
    return a
  else:
    return b

def run(env,color,label,episodes,a,b,is_training=True):
    
    global num
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
        
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 0.2         # 1 = 100% random actions
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)


    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        cnt=0

        while(not terminated and not truncated):
            if is_training and ( rng.random() < epsilon ):
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if  terminated and reward!=b:
                reward=a*10

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
            cnt+=1
            
        if i%10000==0:
            print(str(i)+" "+str(a)+" "+str(b))
           
            

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == b:
            rewards_per_episode[i] = 1
        
    
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[0:(t+1)])
    # plt.clf()
    plt.plot(sum_rewards,color=color,label=label)
    plt.legend()
    # num+=1
    # plt.savefig('f_l4x41'+str(num)+'.png')
    plt.savefig('fl_cumulative.png')



   





def run_env(color,a,b):
    # num=0
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    env = TransformReward(env, lambda r:rewardChange(r,a,b))
    run(env=env,color=color,label="("+str(a)+","+str(b)+")",a=a,b=b,episodes=100000)




if __name__ == '__main__':
    run_env((19/250,106/250,235/250),0,1)  
    run_env((19/250,150/250,200/250),0,5)  
    run_env((19/250,180/250,180/250),0,10)  
    run_env((19/250,210/250,100/250),0,50)  
    run_env((19/250,235/250,68/250),0,100)  
    run_env((191/250,110/250,6/250),-1,0)
    run_env((191/250,75/250,50/250),-1,1)
    run_env((191/250,50/250,126/250),-1,20)
    run_env((191/250,25/250,180/250),-10,0)
    run_env((178/250,191/250,6/250),-10,1)
    run_env((191/250,6/250,250/250),-50,0)
    run_env((191/250,20/250,100/250),-50,1)
    run_env((2/250,10/250,20/250),-200,1)




    # plt.legend()
    # plt.savefig('f_l4x41.png')
