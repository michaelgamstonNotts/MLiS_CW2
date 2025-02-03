import numpy as np
import math
import copy
import os
from random import shuffle

class Card():
    """A Class for each card
    """
    
    def __init__(self, suit : str, type : int, value : int) -> None :
        """Construtor that initiates cards with a value

        Args:
            suit (String): title of the suit
            number (Int): title of the card 
            value (Int): value of the card
        """
        self.suit = suit
        self.type = type
        self.value = value
        
    def change_value(self, new_value : int) -> None:
        """changes the value of a card. 
        for the specail case when an ace is present in the hard 
        and the hand goes over the value of 21

        Args:
            new_value (int): the new value 
        """
        self.value = new_value
        
    def __str__(self):
        return f'{self.type}-{self.suit}-{self.value}'
        
class Deck(): 
    """A class for deck management and manipulation 
    """
    
    def __init__(self) -> None:
        """constructor that creates a new deck and shuffles it  
        """
        
        self.deck = []
        
    def generate_deck(self) -> None:
        """creates a 52 card deck 
        """
        self.deck = []
        suits = ['Diamonds', 'Hearts', 'Clubs', 'Spades']
        card_types = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'Jack':10, 'Queen':10, 'King':10, 'Ace':11}
        for suit in suits:
            for card in card_types.keys():
                self.deck.append(Card(suit, card, card_types[card]))
    
    def shuffle(self) -> None:
        """shuffles the deck
        """
        
        shuffle(self.deck)
    
    def get_cards(self) -> list: 
        """returns a list of shuffled cards

        Returns:
            list: 52 cards
        """
        self.generate_deck()
        self.shuffle()
        return self.deck

class Agent():
    """
    A class for the agent that plays blackjack 
    """
    
    def __init__(self, playable_episodes : int):
        self.playable_episodes = playable_episodes #decreases as episodes deplete (used for infinite agent)
        self.total_iter = playable_episodes #constant iteration count
        
        self.unused_ace = 0 #flag to track if hand has an unused ace
        self.hand = [] #list of cards in current hand 
        self.score = 0 #current hand score 
        #self.cumulative_reward = 0 #cumulative reward over all episodes (not currently tracked)

        #algorithm hyperparameters
        self.alpha = 0.1
        self.epislon = 0.1
        self.gamma = 1
        self.total_iter = playable_episodes #
        
        #degrading alpha hyperparameters 
        self.min_alpha = 0.001
        self.max_alpha = 1.0
        self.alpha_decay_rate = 0.01    
        self.alpha = self.max_alpha # Starting point 

        #Tracking lists
        self.alpha_tracking = []
        self.sumOfHand_tracking = []
        
    def update_tracking(self, is_infinite=True): 
        
        #! ??????
        if is_infinite == True:
            self.alpha_tracking.append(self.alpha)
            self.sumOfHand_tracking.append(self.score)
        else:
            self.alpha_tracking.append(self.alpha)
            self.sumOfHand_tracking.append(self.score)

    def save_tracking(self, is_infinite=True):
        if not os.path.exists('tracking/'):
            os.makedirs('tracking/')
       
        if is_infinite == True:
            np.save("tracking/alpha_track_infinite.npy",self.alpha_tracking)
            np.save("tracking/hand_sum_track_infinite.npy",self.sumOfHand_tracking)
        else:
            np.save("tracking/alpha_track_finite.npy",self.alpha_tracking)
            np.save("tracking/hand_sum_track_finite.npy",self.sumOfHand_tracking)
    
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
    
    def __init__(self, episodes : int) -> None:
        super().__init__(episodes)
        self.q_table_infinite = np.zeros([19,2,2])  #Q-table
        self.policy = None     
    
               
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
        
        #delta, take points from reward when we regress states from loosing an ace 
        delta = 0
        
        #find the state the algorithm was in before the ace changed value and the algorithm switched values 
        #this is to allow correct switching between the no ace and ace side of the q -table
        if used_an_ace: 
            old_state+=10
            delta = old_state**2 - self.score**2
            
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
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp(-(1/(self.alpha_decay_rate*self.total_iter))*(self.total_iter - self.playable_episodes))
    
        
        #Bellman eqaution, used to calculate new Q-values in the Q-table,
        self.q_table_infinite[old_state-2][self.unused_ace][action] = \
                    old_state_value + self.alpha*((reward + self.gamma*max_future_value - old_state_value) - delta) 

        
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
            #Get q value for both hit and stick
            stick_q = self.q_table_infinite[self.score-2][self.unused_ace][0]
            hit_q = self.q_table_infinite[self.score-2][self.unused_ace][1]
           
            if stick_q == hit_q: #If stick and hit are equal, 
                action = np.random.randint(0,2) #pick random move.
            else: #Otherwise, run Epsilon-greedy algorithm to pick the next move.
                if self.epislon > np.random.random():
                    action = np.random.randint(0,2)
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
                self.policy[s_index][u_index] = int(np.argmax(unused_ace))
                
        np.save('infinite_policy.npy', self.policy)


class Finite_agent(Agent):
    """
    An agent to play the finite version of the game

    Args:
        Agent (Agent): The parent class 
    """
    
    def __init__(self, episodes : int, toggle_selective_policy = False):
        super().__init__(episodes)
        self.q_table_finite = np.zeros([19,10,2,2]) # q-table
        self.state_updated_tracker = np.zeros([19,10,2]) #track how many update each state gets to track convergence 
        self.policy = None # empty policy 
        self.card_tracker = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0} # card tracking for calculating probabilities (filled during play through)
        self.loss_state = 0 # variable to track the calculate probabilty of lossing
        self.toggle_selective_policy = toggle_selective_policy #used as a boolean in save_tables (look in function description for further explination)
        self.episodes = episodes #used to track the amount of episodes
       
               
    def calculate_probability_of_loss(self) -> None:
        """This calculates the probability that the next card will make the agent lose
        """
        
        total_number_cards = sum(self.card_tracker.values())
        
        #find the minimum score that will make the agent lose 
        loss_value = 21 - self.score
        
        #find the number of cards that equal or exceed that minimum score, devide by the total number of cards 
        if loss_value < 11:
            numerator = sum(list(self.card_tracker.values())[loss_value:])
        else: 
            numerator = 0
        
        #calculate percentage of lossing, using 10% bins
        percentile = int(round((numerator / total_number_cards), 2)*100)
        if percentile >= 90: 
            self.loss_state = 9 
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
        #calculate loss probability 
        self.calculate_probability_of_loss()
        
        #when training
        if training: 
            #get q value for hit and stick
            stick_q = self.q_table_finite[self.score-2][self.loss_state][self.unused_ace][0]
            hit_q = self.q_table_finite[self.score-2][self.loss_state][self.unused_ace][1]

            
            if stick_q == hit_q: #If stick and hit are equal, 
                action = np.random.randint(0,2) #pick random move.
            else: #Otherwise, run Epsilon-greedy algorithm to pick the next move.
                if self.epislon > np.random.random():
                    action = np.random.randint(0,2)
                else:
                    action = np.argmax(self.q_table_finite[self.score-2][self.loss_state][self.unused_ace])
            
            #return 'hit' or 'stick'
            return action_int_to_str[action]
        
        else: 
            #when not training
            try:
                #if the policy variable is empty fill it with pre-trained table 
                if type(self.policy) != np.ndarray: 
                    self.policy = np.load('finite_policy.npy')
            except FileNotFoundError as e:
                print(e)
                raise RuntimeError('Training required to create policy table.')
            
            #return 'hit' or 'stick'
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
        
        delta = 0
        #find the state the algorithm was in before the ace changed value and the algorithm switched values 
        #this is to allow correct switching between the no ace and ace side of the q -table
        if used_an_ace: 
            old_state+=10
            delta = old_state**2 - self.score**2
            
            
        old_state_value = self.q_table_finite[old_state-2][self.loss_state][self.unused_ace][action] 
        
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
                    max_future_value = np.amax(self.q_table_finite[new_state-2][self.loss_state][0][action])
                else: 
                    #Otherwise, get the max future value from the same side of the Q-table.
                    max_future_value = np.amax(self.q_table_finite[new_state-2][self.loss_state][self.unused_ace][action])
        
        #Recalculate alpha,
        #self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp (- self.alpha_decay_rate * self.episode )
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp(-(1/(self.alpha_decay_rate*self.total_iter))*(self.total_iter - self.episodes))
        #bellman eqaution 
        self.q_table_finite[old_state-2][self.loss_state][self.unused_ace][action] = \
            old_state_value + self.alpha*((reward + self.gamma*max_future_value - old_state_value) - delta)
        
        #track how many update each state gets to track convergence 
        self.state_updated_tracker[old_state-2][self.loss_state][self.unused_ace] += 1

        
        
    def save_tables(self):
        """
        Saves policy and q-tables and the end of training
        """
        np.save('finite_q_table.npy', self.q_table_finite)
        self.policy = np.zeros([19,10,2])
        
        for s_index, state in enumerate(self.q_table_finite): 
            for p_index, percentage in enumerate(state): 
                for u_index, unused_ace in enumerate(percentage):
                    #When toggle_selective_policy == True, only save state pairs with more than 150 iteration to the policy (as states below this value have not had time to converge )
                    if self.toggle_selective_policy: 
                        
                        if (self.state_updated_tracker[s_index][p_index][u_index] > 150): 
                            self.policy[s_index][p_index][u_index] = np.argmax(unused_ace)
                        else:
                            self.policy[s_index][p_index][u_index] = 2 
                            
                    else:
                        self.policy[s_index][p_index][u_index] = np.argmax(unused_ace)
        
        np.save('finite_policy.npy', self.policy)


       
class Dealer(): 
    """
    A class for the (passive) dealer of the blackjack game.
    The dealer essentially runs the game.
    """
    
    def __init__(self, episodes : int, num_deck = 1, is_infinite = False, training = False, toggle_selective_policy = False) -> None:
        
        self.cards = None # total cards remaing 
        self.is_infinite = is_infinite #varible for infinite or finite game
        self.training = training # if training or not
        self.initail_episode_count = episodes
        self.num_decks = num_deck
        
        #instantiate agent based on is_infinite 
        if self.is_infinite:
            self.player = Infinite_agent(episodes)
        else: 
            self.player = Finite_agent(episodes, toggle_selective_policy)
        
        self.get_decks(self.num_decks)
        
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
        
        #load in cards 
        if num_deck == 1: 
            return
        elif num_deck > 1:
        
            for _ in range(1,num_deck): 
                self.cards = np.concatenate((self.cards, np.array(deck.get_cards())))
        else: 
            raise Exception('Interger above 0 required.')
        
        print('New decks set')
        
        
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
        if is_infinite == True: 
            card = copy.copy(self.cards[card_index])
        else: 
            card = self.cards[card_index]
        
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
                self.player.playable_episodes -= 1 
            stop_condition = self.player.playable_episodes
        
        else: #is finite
            cards_left = len(self.cards)
            if cards_left == 0:
                if self.training:
                    if self.player.episodes != 0:
                        self.player.episodes -= 1
                        self.get_decks(self.num_decks)
                        stop_condition = len(self.cards)
                        return stop_condition
            
            stop_condition = cards_left

        return stop_condition
        
        
    def play_game(self) -> None:
        
        """Loops through the game until the number of cards runs out or the select
        number of episodes are finiished.
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
            
            
            while True: 
                
                
                #check if there are cards to play still
                if len(self.cards) < 1:
                    break 
                
                #check if player has won
                if self.player.score == 21: 
                    break
                
                #check if player looses 
                if self.player.score > 21: 
                    if self.player.unused_ace == 0:
                        break 
                    else:
                        self.player.change_ace_value()
                        
                
                #ask player if they want to hit or stick
                response = self.player.assess(training=self.training)
                print(response)
                
                if response == 'hit':
                    #if hit then ask for a new card and pass it to the player 
                    #if training then hit() will update the q-table
                    self.player.hit(self.hit(is_infinite=self.is_infinite), training=self.training)

                elif response == 'stick':
                    #if stick then stop the game 
                    if self.training:
                        #update q-table if training required 
                        self.player.update_q_table(new_card = None, action = 0)
                    break
            
            self.player.update_tracking(is_infinite=self.is_infinite) 
            #print stats at the end of the hand
            print(f'score {self.player.score}, episodes {self.player.episodes if not self.is_infinite else self.player.playable_episodes}, cards {len(self.cards)}')
            #reset the hand
            self.player.reset_hand()
            print('-------')
            #re-evaulate the stop condition to check if the game progresses 
            stop_condition = self.evaulate_stop_condition(is_infinite=self.is_infinite, decrement_hand=self.is_infinite)
        
        #if training then save the q-table and policy 
        if self.training: 
            print('training complete')
            self.player.save_tables()      
            
        self.player.save_tracking(is_infinite=self.is_infinite)
                    
dealer = Dealer(episodes = 10, num_deck=2, is_infinite=False, training=True, toggle_selective_policy = True)
dealer.play_game()
