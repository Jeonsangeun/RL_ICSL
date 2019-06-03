import numpy as np
import random as rd

class cache_replacement:

    def __init__(self):
        self.Num_file = 7 #파일 수
        self.Num_packet = 2 #파일당 패킷 수
        self.Memory = 4 #메모리 크기
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

    def Zip_funtion(self): #zip 분포 생성
        m = np.sum(np.array(range(1, self.Num_file+1))**(-self.alpha))
        self.Zip_law = (np.array(range(1, self.Num_file+1))**(-self.alpha)) / m

    def reset(self): #reset
        self.BS[range(4)] = range(self.Memory)
        self.state = np.zeros([4, self.F_packet])
        for i in range(4):
            self.state[i][self.BS[i]] = self.point
        self.user_location = np.random.uniform(-100, 100, (1, 2))[0]
        self.cost = 0
        self.count = 0
        self.file_request = np.random.choice(self.Setting, 1001, p=self.Zip_law) #패킷 추출
        state = np.append(cache_replacement.flat(self, self.user_location), self.file_request[0]) #, np.array([0, 0])
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
        return aa * self.F_packet + np.random.choice(np.where(self.state[aa] == self.point)[0])

    def Probabilistic(self, d):
        prob = 1.0 - np.exp(-1 * ((2**self.Transmit_Rate - 1)*d**4) / self.Transmission_Power)
        return prob * 100

    def step(self, action, file, user):
        cost = 0
        reward = 0
        done = False
        for i in range(4):
            if action // self.F_packet == i:
                action_1 = action % self.F_packet
                d = cache_replacement.Distance(self, self.BS_Location[i], user)
                if self.state[i][file] == 0:
                    self.state[i][action_1] = 0
                    while np.random.rand(1) < self.M_S_error:
                        cost += self.Macro_BS
                        reward -= self.Macro_BS
                    cost += self.Macro_BS
                    reward -= self.Macro_BS
                    self.state[i][file] = self.point
                    while cache_replacement.error_rate(self, d) == 0:
                        cost += self.Small_BS
                        reward -= self.Small_BS
                    cost += self.Small_BS
                    reward -= self.Small_BS
                    self.count += 1
                elif self.state[i][file] == self.point:
                    while cache_replacement.error_rate(self, d) == 0:
                        cost += self.Small_BS
                        reward -= self.Small_BS
                    cost += self.Small_BS
                    reward -= self.Macro_BS
                    self.count += 1
                elif self.state[i][action_1] == 0:
                    print("error")
                else:
                    while np.random.rand(1) < self.M_S_error:
                        cost += self.Macro_BS
                        reward -= self.Macro_BS
                    cost += self.Macro_BS
                    reward -= self.Macro_BS
                    while cache_replacement.error_rate(self, d) == 0:
                        cost += self.Small_BS
                        reward -= self.Small_BS
                    cost += self.Small_BS
                    reward -= self.Small_BS
                    self.count += 1

        if self.count % self.Num_packet == 0:
            file = self.file_request[self.count // self.Num_packet]
            user = np.random.uniform(-100, 100, (1, 2))[0]
        else:
            file += 1

        new_state = cache_replacement.flat(self, user)
        new_state = np.append(new_state, file)

        self.cost += cost
        if self.count == 1000 * self.Num_packet:
            done = True

        return new_state, reward, done, file, user

    def flat(self, user):
        prob = np.zeros([4])
        d = np.zeros([4])
        for i in range(4):
            d[i] = np.array(cache_replacement.Distance(self, self.BS_Location[i], user))
            prob[i] = np.array(cache_replacement.Probabilistic(self, d[i]))
        result = np.reshape(self.state, [1, 4 * self.F_packet])
        result = np.hstack([prob, result[0]])
        return result

    def Q_fun(self, Q):
        result = np.reshape(self.state, [1, 4 * self.F_packet])
        Q[np.where(result[0] == 0)[0]] = -100000000000000000000
        return Q

    def print(self):
        print(np.where(self.state[0] == self.point)[0])
        print(np.where(self.state[1] == self.point)[0])
        print(np.where(self.state[2] == self.point)[0])
        print(np.where(self.state[3] == self.point)[0])
