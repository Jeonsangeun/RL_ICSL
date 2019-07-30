import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 14 #파일 수
        self.Num_packet = 4 #파일당 패킷 수
        self.Memory = 8 #모리 크기
        self.x_max = 100 #이용자 최대 위치(횡축)
        self.y_max = 100 #이용자 최대 위치(종축)
        self.F_packet = self.Num_file * self.Num_packet #총 패킷 수
        self.alpha = 0.8#zip 알파
        self.Transmission_Power = 10**9 #송신전력
        self.user_location = np.zeros([2])  #유저의 위치
        self.BS = np.zeros([4, self.Memory], dtype=int) #BS 송신소 상태
        self.BS[range(4)] = range(self.Memory) #가장 유명한 패킷 0~3을 담은 상태
        self.BS_Location = np.array([[(-1 * self.x_max / 2), (self.y_max / 2)],
                                    [(self.x_max / 2), (self.y_max / 2)],
                                    [(-1 * self.x_max / 2), (-1 * self.y_max / 2)],
                                    [(self.x_max / 2), (-1 * self.y_max / 2)]]) #BS0의 위치
        self.state = np.zeros([4, self.F_packet]) #state 설정
        self.Transmit_Rate = 1.0 #전송률
        self.M_S_error = 0.1 #Macro->Small 에러율
        self.Macro_BS = 10 #Macro->Small cost
        self.Small_BS = 5 #Small->Macro cost
        self.count = 0 #패킷 리퀘스트 카운트
        self.cost = 0 #1episode 당 cost
        self.point = 10 #state 설정값
        self.Zip_law = [] #zip 분포
        self.Setting = tuple(range(0, self.F_packet, self.Num_packet)) #zip 분포 파일
        self.file_request = []
        self.error = 0

    def Zip_funtion(self): #zip 분포 생성
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m

    def reset(self): #reset
        self.BS[range(4)] = range(self.Memory)
        self.state = np.zeros([4, self.F_packet])
        for i in range(4):
            self.state[i][self.BS[i]] = self.point
        self.user_location = np.random.uniform(-1*self.x_max, self.x_max, (1, 2))[0]
        self.cost = 0
        self.count = 0
        self.file_request = np.random.choice(self.Setting, 1001, p=self.Zip_law) #패킷 추출
        state = np.append(self.flat(self.user_location), self.file_request[0]) #, np.array([0, 0])
        return state

    def error_rate(self, d): #error_rate 0 : 실패 1 : 성공
        w = np.random.exponential(1)
        channel = d**(-4) * w
        if self.Transmit_Rate > np.log2(1 + channel * self.Transmission_Power):
            return 0
        else:
            return 1

    def Distance(self, x, y): #좌표간의 거리
        return np.sqrt(np.sum((x - y)**2))

    def random_action(self):
        aa = rd.randrange(4)
        bb = rd.randrange(self.Memory)
        return 4 * aa + bb

    def Probabilistic(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**4) / self.Transmission_Power)
        return prob * 100

    def step(self, action, file, user):
        cost = 0
        reward = 0
        done = False
        for i in range(4):
            if action // (self.Memory) == i:
                action_1 = action % (self.Memory)
                d = self.Distance(self.BS_Location[i], user)
                if self.state[i][file] == self.point:
                    n = np.random.geometric(p=(1 - self.Probabilistic(d)/100))
                    cost += n * self.Small_BS
                    reward -= n * self.Small_BS
                    self.count += 1
                else:
                    self.state[i][self.BS[i][action_1]] = 0
                    m = np.random.geometric(p=(1-self.M_S_error))
                    cost += m * self.Macro_BS
                    reward -= m * self.Macro_BS
                    self.state[i][file] = self.point
                    n = np.random.geometric(p=(1 - self.Probabilistic(d)/100))
                    cost += n * self.Small_BS
                    reward -= n * self.Small_BS
                    self.count += 1

        if self.count % self.Num_packet == 0:
            file = self.file_request[self.count // self.Num_packet]
            user = np.random.uniform(-1 * self.x_max, self.x_max, (1, 2))[0]
        else:
            file += 1
        self.apply()
        new_state = self.flat(user)
        new_state = np.append(new_state, file)

        self.cost += cost
        if self.count == 200 * self.Num_packet:
            done = True

        return new_state, reward, done, file, user

    def flat(self, user):
        prob = np.zeros([4])
        d = np.zeros([4])
        for i in range(4):
            d[i] = np.array(self.Distance(self.BS_Location[i], user))
            prob[i] = np.array(self.Probabilistic(d[i]))
        result = np.reshape(self.state, [1, 4 * self.F_packet])
        result = np.hstack([result[0], prob])
        return result

    def print(self):
        print(self.BS[0])
        print(self.BS[1])
        print(self.BS[2])
        print(self.BS[3])

    def apply(self):
        self.BS[0] = np.where(self.state[0] == self.point)[0]
        self.BS[1] = np.where(self.state[1] == self.point)[0]
        self.BS[2] = np.where(self.state[2] == self.point)[0]
        self.BS[3] = np.where(self.state[3] == self.point)[0]
