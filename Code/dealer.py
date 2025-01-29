from card import Deck, Card
import numpy as np
import math
import copy
import os

class Agent():
    """
    A class for the agent that plays blackjack 
    """
    
    def __init__(self, playable_hands : int):
        self.playable_hands = playable_hands #decreases as hands deplete (used for infinite agent)
        self.unused_ace = 0 #flag to track if hand has an unused ace
        self.hand = [] #list of cards in current hand 
        self.score = 0 #current hand score 
        self.cumulative_reward = 0 #cumulative reward over all hands (not currently tracked)

        #algorithm hyperparameters
        self.alpha = 0
        self.epislon = 0.1
        self.gamma = 1
        self.total_iter = playable_hands
        
        self.min_alpha = 0.01
        self.max_alpha = 1.0
        self.alpha_decay_rate = 0.2 
        self.episode = self.total_iter - self.playable_hands
        
        self.alpha_tracking = np.zeros(playable_hands)
        self.sumOfHand_tracking = np.zeros(playable_hands)
        
    def update_tracking(self): 
        self.alpha_tracking[self.total_iter - self.playable_hands] = self.alpha
        self.sumOfHand_tracking[self.total_iter - self.playable_hands] = self.score

    def save_tracking(self):
        if not os.path.exists('tracking/'):
            os.makedirs('tracking/')
        
        np.save("tracking/alpha_tracking_data.npy",self.alpha_tracking)
        np.save("tracking/hand_sum_tracking_data.npy",self.sumOfHand_tracking)
    
    #to be overwritten by child classes 
    def update_q_table(self):
        raise NotImplementedError('update_q_table not implemented')

    #to be overwritten by child classes 
    def assess(self):
        raise NotImplementedError('assess not implemented')
        
    def check_for_unused_ace(self) -> None:
        """
        Checks if there is an unused ace present in the hand and 
        sets the unused_ace flag to one if ace present.
        """
        unused_ace_ = [card for card in self.hand if (card.type == 'Ace') and (card.value == 11)]
        if len(unused_ace_) > 0:
            self.unused_ace = 1
                
    def change_ace_value(self) -> None:
        """
        Finds the first unused ace in hand and drops its value down to 1.
        """
        for card in self.hand:
            if (card.type == 'Ace') and (card.value == 11):
                card.change_value(1)
                self.score -= 10
                self.unused_ace = 0
        print('changed ace value')
        
    def hit(self, new_card : Card, training=False) -> None:
        """
        Recieves a new card and adds it to the hand.
        if training == true, then it also updates the Q - table.

        Args:
            new_card (Card): new card received from the dealer 
            training (bool, optional):  Defaults to False.
        """
        #calculates the new total hand value (stores locally)
        new_score = new_card.value + self.score
        
        if training == True:
            #If the new score either wins or continues the game,
            if new_score <= 21: 
                if new_score == 21: #hits 21, update q table with a win condition.
                    self.update_q_table(new_card = new_card, action = 1, win_case = True)
                else: #below 21, update q table without a win condition.
                    self.update_q_table(new_card = new_card, action = 1)
                    
            #If the new score exceeds the score limit, and the NEW card IS an ace,
            elif new_score > 21 and new_card.type == 'Ace':
                
                new_card.change_value(1) #Drop ace value from 11 to 1.
                new_score = new_card.value + self.score #Recalculate score.
                
                if new_score == 21: #hits 21, update q table with a win condition.
                    self.update_q_table(new_card = new_card, action = 1, win_case = True)
                if new_score < 21: #below 21, update q table without a win condition.
                    self.update_q_table(new_card = new_card, action = 1)
                    
            #If the new score exceeds the score limit, and the NEW card IS NOT an ace,
            elif new_score > 21 and new_card.type != 'Ace':
                if self.unused_ace: #If there is an unused ace,
                    
                    self.change_ace_value() #Drop value of previous ace.
                    #Keep the flag = 1 while the q-table is updated so the correct side is updated.
                    self.unused_ace = 1
                    self.update_q_table(new_card = new_card, action = 1, win_case = False, used_an_ace = True)
                    self.unused_ace = 0 
                    
                else: #If there is not an unused ace,
                    self.update_q_table(new_card = new_card, action = 1, win_case = False)

        #Adds the new card to the current score and hand. 
        self.score += new_card.value
        self.hand.append(new_card)
        self.check_for_unused_ace()
        
    def reset_hand(self) -> None:
        """
        Reset the hand at the start of a new hand.
        """
        self.hand = []
        self.score = 0
        self.unused_ace = 0
        
    def save_tables(self):
        #To be overwritten by child classes
        raise NotImplementedError('save_tables not implemented')
 

       
class Infinite_agent(Agent):
    """
    A agent to learn the infinite version of black jack usuing q-learning 

    Args:
        Agent (Agent): Parent class 
    """
    
    def __init__(self, hands : int) -> None:
        super().__init__(hands)
        self.q_table_infinite = np.zeros([19,2,2])  #Q-table
        self.policy = None                          #Empty Policy
               
    def update_q_table(self, new_card : Card, action : int, win_case = False, used_an_ace = False) -> None:
        """
        Used to update the the q-table in case of an infinite agent 

        Args:
            new_card (Card): The card receievd from the dealer (None if agent sticks)
            action (int): 0 for stick, 1 for hit
            win_case (bool, optional): Special use case for the new state is determined to be 21. Defaults to False.
            used_an_ace (bool, optional): special use case for if and ace has been decremenetd in hit(). Defaults to False.
        """
    
        if new_card == None: 
            new_card_value = 0
        else:
            new_card_value = new_card.value
        
        #Calculates old state, the new state and the value of the old state 
        old_state = self.score
        new_state = old_state+new_card_value
        
        #delta, take points from reward when we regress from loosing an ace 
        delta = 0
        #fixed aces bug 
        #keep state of ace to be what it was before the change 
        #this is to allow correct switching between the no ace and ace side of the q -table
        if used_an_ace: 
            old_state+=10
            delta = old_state**2 - self.score**2
            #print(old_state, self.score)
            
            
        old_state_value = self.q_table_infinite[old_state-2][self.unused_ace][action] 
        
        if new_state > 21: #If the new score exceeds the score limit,
            reward = 0
            max_future_value = 0
            
        elif new_state <= 21: #If the new score is within the score limit,
            reward = new_state**2 if action == 1 else self.score**2 #calculate score based on move.
            if win_case or action == 0: #If the game is won, or the agent has stuck,
                max_future_value = 0
            elif action == 1: #If the agent hits and does not win,
                if used_an_ace:
                    #If an ace has been used this move, then get the max future value from the No Ace side of the Q-table.
                    max_future_value = np.amax(self.q_table_infinite[new_state-2][0][action])
                else: 
                    #Otherwise, get the max future value from the same side of the Q-table.
                    max_future_value = np.amax(self.q_table_infinite[new_state-2][self.unused_ace][action])
        
        
        #Recalculate alpha,
        #self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp (- self.alpha_decay_rate * self.episode )
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp(-(1/(self.alpha_decay_rate*self.total_iter))*(self.total_iter - self.playable_hands))
    
        
        #Bellman eqaution, used to calculate new Q-values in the Q-table,
        self.q_table_infinite[old_state-2][self.unused_ace][action] = \
                    old_state_value + self.alpha*((reward + self.gamma*max_future_value - old_state_value) - delta) 
        print('Updated Q-table')
        
    def assess(self, training = False) -> str:
        """
        When training, this function returns an action based on an Epsilon-greedy algorithm.
        When not training, this function picks the best action according to the Policy (defined during training).

        Args:
            training (bool, optional): training use case, defaults to False.

        Raises:
            e: file not found error
            RuntimeError: raises error is no policy table has been computed and training is still required

        Returns:
            str: the decided action 
        """
        
        #Maps integer moves to their respective strings.
        action_int_to_str = {0:'stick', 1:'hit'}
        
        if training == True: 
            #get q value for hit and stick       
            #try:
            stick_q = self.q_table_infinite[self.score-2][self.unused_ace][0]
            hit_q = self.q_table_infinite[self.score-2][self.unused_ace][1]
            # except IndexError as e: 
            #     print(e)
            #     print(f'score: {self.score} aces {self.unused_ace}')
            #     print(*self.hand)
            #     raise e
            
            if stick_q == hit_q: #If stick and hit are equal, 
                action = np.random.randint(0,2) #pick random move.
            else: #Otherwise, run Epsilon-greedy algorithm to pick the next move.
                if self.epislon > np.random.random():
                    action = np.argmin(self.q_table_infinite[self.score-2][self.unused_ace])
                else:
                    action = np.argmax(self.q_table_infinite[self.score-2][self.unused_ace])
            return action_int_to_str[action]
        
        if training == False: 
            try: #Load policy file if it is not already defined in self.policy.
                if type(self.policy) != np.ndarray: 
                    self.policy = np.load('infinite_policy.npy')
            except FileNotFoundError as e: #Throw error if no policy file can be found.
                print(e)
                raise RuntimeError('Training required to create policy table.')
            return action_int_to_str[self.policy[self.score-2][self.unused_ace]]
        
    def save_tables(self) -> None:
        """
        Saves policy and q-tables and the end of training
        """
        np.save('infinite_q_table.npy', self.q_table_infinite)
        self.policy = np.zeros([19,2])

        for s_index, state in enumerate(self.q_table_infinite): 
            for u_index, unused_ace in enumerate(state):
                self.policy[s_index][u_index] = np.argmax(unused_ace)
                
        np.save('infinite_policy.npy', self.policy)



class Finite_agent(Agent):
    """
    An agent to play the finite version of the game

    Args:
        Agent (Agent): The parent class 
    """
    
    def __init__(self, hands):
        super().__init__(hands)
        self.q_table_finite = np.zeros([19,10,2,2]) # q-table 
        self.policy = None # empty policy 
        self.card_tracker = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0} # card tracking for calculating probabilities (filled during play through)
        self.loss_state = 0 # variable to track the calculate probabilty of lossing
        self.loss_state_tracker = np.zeros(10) # for debugging, will delete before hand in 
               
    def calculate_probability_of_loss(self) -> None:
        """This calculates the probability that the next card will make the agent loose
        """
        
        #! subject to change 
        
        total_number_cards = sum(self.card_tracker.values())
        #find the minimum score that will make the agent lose 
        #find the number of cards that equal or exceed over the total number of cards 
        loss_value = 21 - self.score
        
        if loss_value < 11:
            numerator = sum(list(self.card_tracker.values())[loss_value:])
        else: 
            numerator = 0
        print(self.card_tracker.values())
        print(f'score {self.score} - loss value {loss_value} - numerator {numerator} - denom {total_number_cards}')
        
        percentile = int(round((numerator / total_number_cards), 2)*100)
        if percentile >= 90: 
            self.loss_state = 9 
            print('yes 90')
        elif percentile >= 80: 
            self.loss_state = 8 
        elif percentile >= 70:
            self.loss_state = 7 
        elif percentile >= 60: 
            self.loss_state = 6
        elif percentile >= 50:
            self.loss_state = 5
        elif percentile >= 40:
            self.loss_state = 4
        elif percentile >= 30: 
            self.loss_state = 3
        elif percentile >= 20:
            self.loss_state = 2
        elif percentile >= 10:
            self.loss_state = 1
        elif percentile >= 0:
            self.loss_state = 0  
             
        
        #self.loss_state = int(round((numerator / total_number_cards), 2)*100)
        print(f'- loss_state {self.loss_state}')
        
    def assess(self, training = False) -> None:
        """
        When training, this function returns an action based on an Epsilon-greedy algorithm.
        When not training, this function picks the best action according to the Policy (defined during training).

        Args:
            training (bool, optional): training use case, defaults to False.

        Raises:
            e: file not found error
            RuntimeError: raises error is no policy table has been computed and training is still required

        Returns:
            str: the decided action 
        """
        
        action_int_to_str = {0:'stick', 1:'hit'}
        # for training only 
        self.calculate_probability_of_loss()
        
        if training: 
            #get q value for hit and stick
            #! remove the try statement before hand in 
            try:
                stick_q = self.q_table_finite[self.score-2][self.loss_state][self.unused_ace][0]
                hit_q = self.q_table_finite[self.score-2][self.loss_state][self.unused_ace][1]
            except IndexError as e: 
                print(e)
                print(f'score: {self.score} aces {self.unused_ace}')
                print(*self.hand)
                raise e
            
            #check if they are equal
            if stick_q == hit_q:
                #choose random action is yes
                action = np.random.randint(0,2)
            else:
                #else run epsilon greedy to find action 
                if self.epislon > np.random.random():
                    action = np.argmin(self.q_table_finite[self.score-2][self.loss_state][self.unused_ace])
                else:
                    action = np.argmax(self.q_table_finite[self.score-2][self.loss_state][self.unused_ace])
            
            return action_int_to_str[action]
        
        else: 
            try:
                if type(self.policy) != np.ndarray: 
                    self.policy = np.load('infinite_policy.npy')
            except FileNotFoundError as e:
                print(e)
                raise RuntimeError('Training required to create policy table.')
            
            return action_int_to_str[self.policy[self.score-2][self.loss_state][self.unused_ace]]
        
    def update_q_table(self, new_card : Card, action : int, win_case = False, used_an_ace = False):
        """
        Used to update the the q-table in case of an finite agent 

        Args:
            new_card (Card): The card receievd from the dealer (None if agent sticks)
            action (int): 0 for stick, 1 for hit
            win_case (bool, optional): Special use case for the new state is determined to be 21. Defaults to False.
            used_an_ace (bool, optional): special use case for if and ace has been decremenetd in hit(). Defaults to False.
        """
        
        if new_card == None: 
            new_card_value = 0
        else:
            new_card_value = new_card.value
        
        old_state = self.score
        new_state = old_state+new_card_value
        
        #fixed aces bug 
        #keep state of ace to be what it was before the change 
        #this is to allow correct switching  between the no ace and ace side of the q -table
        if used_an_ace: 
            old_state+=10
            
        old_state_value = self.q_table_finite[old_state-2][self.loss_state][self.unused_ace][action] 
        
        if new_state > 21:
            reward = 0
            max_future_value = 0
        
        else:
            reward = new_state**2 if action else self.score**2 
            if win_case or (action == 0): 
                max_future_value = 0
            elif action == 1:
                if used_an_ace:
                    max_future_value = np.amax(self.q_table_finite[new_state-2][self.loss_state][0][action])
                else:
                    max_future_value = np.amax(self.q_table_finite[new_state-2][self.loss_state][self.unused_ace][action])
        
        #! need to think of how to do degrading alpha for finite cards
        #self.alpha = 0.3/(math.exp(self.playable_hands/len(self.cards)))
        #bellman eqaution 
        self.q_table_finite[old_state-2][self.loss_state][self.unused_ace][action] = \
            old_state_value + self.alpha*(reward + self.gamma*max_future_value - old_state_value)
            
        self.loss_state_tracker[self.loss_state] += 1
        print(f'q-table updated at state {old_state-2},{self.loss_state},{self.unused_ace},{action}')
        
    def save_tables(self):
        """
        Saves policy and q-tables and the end of training
        """
        np.save('finite_q_table.npy', self.q_table_finite)
        self.policy = np.zeros([19,10,2])
        #! think of some better names here 
        for s_index, state in enumerate(self.q_table_finite): 
            for p_index, percentage in enumerate(state): 
                for u_index, unused_ace in enumerate(percentage):
                    self.policy[s_index][p_index][u_index] = np.argmax(unused_ace)
                
        np.save('finite_policy.npy', self.policy)


       
class Dealer(): 
    """
    A class for the (passive) dealer of the blackjack game.
    The dealer essentially runs the game.
    """
    
    def __init__(self, hands : int, is_infinite = False, training = False) -> None:
        
        self.cards = None # total cards remaing 
        self.is_infinite = is_infinite #varible for infinite or finite game
        self.training = training # if training or not
        
        #instantiate agent based on is_infinite 
        if self.is_infinite:
            self.player = Infinite_agent(hands)
        else: 
            self.player = Finite_agent(0)
        
        
          
    def get_decks(self, num_deck : int) -> None: 
        """Collects the allotted number of decks 

        Args:
            num_deck (int): the required number of decks 
        Raises:
            Exception: if a value below 1 is entered an exception is thrown
        """
        deck = Deck() #Deck class.
        #instantiate array the size of the number of cards to be recieved
        self.cards = np.array(deck.get_cards())
        
        #if finite agent then set up card tracking 
        if self.is_infinite == False: 
            for card_type in list(self.player.card_tracker.keys())[:-1]: 
                self.player.card_tracker[card_type] = 4*num_deck
            self.player.card_tracker[10] = 16*num_deck
            print('card tracker')
            print(self.player.card_tracker)
        
        #load in cards 
        if num_deck == 1: 
            return
        elif num_deck > 1:
            for _ in range(1,num_deck):
                self.cards = np.concatenate((self.cards, np.array(deck.get_cards())))
        else: 
            raise Exception('Interger above 0 required.')
        
        
        
    def hit(self, is_infinite = False) -> Card:
        """
        Gives the agent a random card when requested. 
        Either deletes the card from the deck for finite situation,
        or keeps the card in the deck for the infinite.

        Args:
            is_infinate_cards (bool, optional): argument to decide between is_infinite and finate . Defaults to False.

        Returns:
            Card: the selected card 
        """
        
        #randomly select a card 
        card_index = np.random.randint(0, len(self.cards))
        #copy the card (no directly referenced to avoid conflict in inifinite version)
        card = copy.copy(self.cards[card_index])
        
        #if finite then delete card from cards otherwise keep it
        if is_infinite == False: 
            self.cards = np.delete(self.cards, card_index)
            if card.type == 'Ace':
                self.player.card_tracker[1] -= 1
            else:
                self.player.card_tracker[card.value] -= 1 
            
        return card
    
    def evaulate_stop_condition(self, is_infinite = False, decrement_hand = False) -> int:
        """checks how far the game has progressed

        Args:
            is_infinite (bool, optional): this is used to decied which stop condition to use. Defaults to False.
            decrement_hand (bool, optional): required in the case that of infinite cards. Defaults to False.

        Returns:
            int: _description_
        """
        if is_infinite: 
            if decrement_hand:
                self.player.playable_hands -= 1 
            
            stop_condition = self.player.playable_hands
        else: 
            stop_condition = len(self.cards)
            
        return stop_condition
        
        
    def play_game(self) -> None:
        
        """Loops through the game until the number of cards runs out or the select
        number of hands are finiished.
        """
        #find which sstop condition to use 
        stop_condition = self.evaulate_stop_condition(is_infinite=self.is_infinite)

        while(0 < stop_condition):
            
            # give player a card
            first_card = self.hit(is_infinite=self.is_infinite)
            #manually add card info to agent 
            self.player.score = first_card.value 
            self.player.hand.append(first_card)
            self.player.check_for_unused_ace()

            print('\nfirst hand given ----------------------------------------------')
            while True: 
                print(f'round begins with: score {self.player.score}, aces {self.player.unused_ace}')
                print(*self.player.hand)
                
                #check if there are cards to play still
                if len(self.cards) < 1:
                    break 
                
                #check if player has won
                if self.player.score == 21: 
                    break
                
                #check if player looses 
                if self.player.score > 21: 
                    if self.player.unused_ace == 0:
                        print('player loses')
                        break 
                    else:
                        #! need to update q-value
                        self.player.change_ace_value()
                        print('got to over 21, changed ace value')
                
                #ask player if they want to hit or stick
                response = self.player.assess(training=self.training)
                print(response)
                
                if response == 'hit':
                    #if hit then ask for a new card and pass it to the player 
                    #if training then hit() will update the q-table
                    self.player.hit(self.hit(is_infinite=self.is_infinite), training=self.training)
                    print('player hits')
                elif response == 'stick':
                    #if stick then stop the game 
                    if self.training:
                        #update q-table if training required 
                        print('update from stick')
                        self.player.update_q_table(new_card = None, action = 0)
                    print(f'player sticks with score {self.player.score} and reward {self.player.cumulative_reward}')
                    break
            
            self.player.update_tracking() 
            #print stats at the end of the hand
            print(f'score {self.player.score}, hands {self.player.playable_hands}, cards {len(self.cards)}')
            #reset the hand
            self.player.reset_hand()
            #re-evaulate the stop condition to check if the game progresses 
            stop_condition = self.evaulate_stop_condition(is_infinite=self.is_infinite, decrement_hand=self.is_infinite)
        
        #if training then save the q-table and policy 
        if self.training: 
            print('epdisode of training complete')
            self.player.save_tables()      
            
        #printing stats (not needed by helpful for debugging)
        print(f'Game ends with {self.player.score} score, and {self.player.cumulative_reward} reward,\n Hands: {self.player.playable_hands}, card count: {len(self.cards)}')
        if self.is_infinite == False: 
            for x,i in enumerate(self.player.loss_state_tracker):
                print(f'{x} - {i}')


        self.player.save_tracking()
        
        
            
dealer = Dealer(hands = 50000, is_infinite=True, training=True)
dealer.get_decks(1000)
dealer.play_game()
