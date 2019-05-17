import numpy as np

class Black:
    def __init__(self):
        self.deck = []
        self.Dealer = []
        self.Player = []
        self.Total_count = 0
        self.state = np.array([0, 0, 0])
        self.Usable_Ace = 0

    def deck_shuffle(self):
        aa = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.deck = np.array(aa * 4)
        np.random.shuffle(self.deck)
        return self.deck

    def state_make(self, Dealer, Player):
        self.Usable_Ace = 0
        aa = Black.Count_Point(self, Dealer) - 2
        bb = Black.Count_Point(self, Player) - 12
        if 'A' in Player:
            self.Usable_Ace = 1
        state = tuple([aa, bb, self.Usable_Ace])
        return state

    def reset(self):
        self.deck = Black.deck_shuffle(self)
        self.Total_count = 0
        self.Dealer = []
        self.Player = []
        self.Dealer.append(self.deck[0])
        self.Player.append(self.deck[2])
        self.Player.append(self.deck[3])
        while Black.Count_Point(self, self.Player) < 12:
            self.Player.append(self.deck[self.Total_count + 4])
            self.Total_count += 1
        self.state = Black.state_make(self, self.Dealer, self.Player)
        return self.state

    def random(self):
        return np.random.choice([0, 1])

    def Count_Point(self, Cards):
        Total_count = 0
        Ace_count = 0
        for i in Cards:
            if (i != 'A'):
                Total_count += int(i)
            else:
                Ace_count += 1
        while Ace_count != 0:
            if (Total_count <= 10):
                Total_count += 11
                Ace_count -= 1
            else:
                Total_count += 1
                Ace_count -= 1
        return Total_count

    def step(self, action):
        reward = 0
        done = False
        Nextstate = np.array([0, 0, 0])
        if action == 0:
            Nextstate = Black.state_make(self, self.Dealer, self.Player)
            self.Dealer.append(self.deck[1])
            Black.Dealer_AI(self)
            if Black.Count_Point(self, self.Dealer) < Black.Count_Point(self, self.Player) \
                    or Black.Count_Point(self, self.Dealer) > 21:
                reward += 1
            elif Black.Count_Point(self, self.Dealer) == Black.Count_Point(self, self.Player):
                reward += 0
            else:
                reward -= 1
            done = True
        if action == 1:
            Nextstate = Black.state_make(self, self.Dealer, self.Player)
            self.Player.append(self.deck[self.Total_count + 4])
            if Black.Count_Point(self, self.Player) > 21:
                self.Dealer.append(self.deck[1])
                reward -= 1
                done = True
            else:
                reward += 0
                done = False
                Nextstate = Black.state_make(self, self.Dealer, self.Player)
                self.Total_count += 1
        return Nextstate, reward, done

    def Dealer_AI(self):
        while Black.Count_Point(self, self.Dealer) < 17:
            self.Dealer.append(self.deck[self.Total_count + 4])
            self.Total_count += 1
