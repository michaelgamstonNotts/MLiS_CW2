from card import Deck, Card
import numpy as np
import math
import copy
#hello
class Agent():
    """A class for the agent that plays blackjack 
    will contain q learning 
    """
    
    def __init__(self, playable_hands : int):
        self.playable_hands = playable_hands #decreases as hands depleat (used for infinite agent)
        self.unused_ace = 0 #flag to track if hand has an unused ace
        self.hand = [] #list of cards in current hand 
        self.score = 0 #current hand score 
        self.cumulative_reward = 0 #cumulative reward over all hands (not currently tracked)


        #algorithm hyerparameters
        self.alpha = 1.0
        self.epislon = 0.3
        self.gamma = 0.7
        self.total_iter = playable_hands
        #more hyperparameters
        self.min_alpha = 0.01
        self.max_alpha = 1.0
        self.decay_rate = 0.045
        self.episode = self.total_iter - self.playable_hands
    
    #to be overwritten by child classes 
    def update_q_table(self):
        raise NotImplementedError('update_q_table not implemented')

    #to be overwritten by child classes 
    def assess(self):
        raise NotImplementedError('assess not implemented')
        
    
    def check_for_unused_ace(self) -> None:
        """Checks if there is an unused ace present in the hand 
        sets the unused_ace flag to one if ace present 
        """
        aces = [card for card in self.hand if (card.type == 'Ace') and (card.value == 11)]
        if len(aces) > 0:
            self.unused_ace = 1
                
    def change_ace_value(self) -> None:
        """finds the first unused ace in hand and changes it's value down to 1
        """
        for card in self.hand:
            if (card.type == 'Ace') and (card.value == 11):
                card.change_value(1)
                self.score -= 10
                self.unused_ace = 0
        print('changed ace value')
        
    def hit(self, new_card : Card, training=False) -> None:
        """recieves new card and adds it to the hand 
        if training == true then it also updates the Q - table

        Args:
            new_card (Card): new card received from the dealer 
            training (bool, optional):  Defaults to False.
        """
        #calculates the new total hand value (stores locally)
        new_score = new_card.value + self.score
        #checks if training and if the new total is below 21 
        
        if training:
            if (new_score < 22): 
                #if new total is 21 then give the update q -table function the win_case boolean 
                if new_score == 21: 
                    print('yes?')
                    self.update_q_table(new_card = new_card, action = 1, win_case = True)
                else: 
                #if new score 20 or below then update the q-table accordingly
                    self.update_q_table(new_card, 1)
                    
            #if training and score is 22 and over, and the new card is an ace 
            elif new_card.type == 'Ace':
                
                #decrement value of ace 
                new_card.change_value(1)
                #recalculate new_score 
                new_score = new_card.value + self.score
                #if new_score is 21 then game over max reward 
                if new_score == 21:
                    self.update_q_table(new_card = new_card, action = 1, win_case = True)
                #other wise proceed as normal
                if new_score < 21:
                    self.update_q_table(new_card = new_card, action = 1)
                    
            # if training and new score is 22 and over 
            else:
                #check if the hand has an unused ace
                if self.unused_ace: 
                    #if we have one, find it and decrement it's value
                    self.change_ace_value()
                    #keep the flag untill after the q-table is updated so we update the correct side
                    self.unused_ace = 1
                    self.update_q_table(new_card = new_card, action = 1, win_case = False, used_an_ace = True)
                    #unassign flag 
                    self.unused_ace = 0 
                else:
                    #if not ace then update the q-table accordingly 
                    self.update_q_table(new_card = new_card, action = 1, win_case = False)
            
        # add the cards to our hand and update score check for unused ace 
        self.score += new_card.value
        self.hand.append(new_card)
        self.check_for_unused_ace()
        
    def reset_hand(self) -> None:
        """resest the hand at the start of a new hand 
        """
        self.hand = []
        self.score = 0
        self.unused_ace = 0
        
    #to be overwritten by child classes
    def save_tables(self):
        raise NotImplementedError('save_tables not implemented')
        
class Infinite_agent(Agent):
    """
    A agent to learn the infinite version of black jack usuing q-learning 

    Args:
        Agent (Agent): Parent class 
    """
    
    def __init__(self, hands : int) -> None:
        super().__init__(hands)
        self.q_table_infinite = np.zeros([19,2,2]) # q-table
        self.policy = None # empty policy 
               
    def update_q_table(self, new_card : Card, action : int, win_case = False, used_an_ace = False) -> None:
        """used to update the the q-table in case of an infinite agent 

        Args:
            new_card (Card): the card receievd from dealer (is None if agent sticks)
            action (int): 0 for stick 1 for hit
            win_case (bool, optional): special use case for the new state is determined to be 21. Defaults to False.
            used_an_ace (bool, optional): special use case for if and ace has been decremenetd in hit(). Defaults to False.
        """
        
        #check if there is a new card or not
        if new_card == None: 
            new_card_value = 0
        else:
            new_card_value = new_card.value
        
        #calculates old state, the new state and the value of the old state 
        old_state = self.score
        new_state = old_state+new_card_value
        
        #fixed aces bug 
        #keep state of ace to be what it was before the change 
        #this is to allow correct switching  between the no ace and ace side of the q -table
        if used_an_ace: 
            old_state+=10
            
        old_state_value = self.q_table_infinite[old_state-2][self.unused_ace][action] 
        
        #check if the new state is over 21
        if new_state > 21:
            #if yes then the agent recieves no reward and no future rewards 
            reward = 0
            max_future_value = 0
        #if new state within bounds 
        else:
            #calculate the new reward based on action (if 0 reward = current state**2 isf 1 rewards = new state**2)
            reward = new_state**2 if action == 1 else self.score**2 
            if win_case or (action == 0): 
                #if the agent has got 21 or sticks then there is no future reward to get 
                max_future_value = 0
            elif action == 1:
                #if the agents hits and does not get 21 
                if used_an_ace:
                    #if special ace use case activate (and unused ace flad still set to 1) get max future reward from the no ace side of the table 
                    max_future_value = np.amax(self.q_table_infinite[new_state-2][0][action])
                else:
                    #other wise get it from the same side of the table
                    max_future_value = np.amax(self.q_table_infinite[new_state-2][self.unused_ace][action])
        

        #decreasing alpha
        self.alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * math.exp (- self.decay_rate * self.episode )

        #bellman eqaution 
        self.q_table_infinite[old_state-2][self.unused_ace][action] = \
            old_state_value + self.alpha*(reward + self.gamma*max_future_value - old_state_value)
            
        print('q-table updated')
        
    def assess(self, training = False) -> str:
        """when training this returns an action based on and epsilon greedy algorithm 
        when not training this picks the best action according to the q-table

        Args:
            training (bool, optional): training use case. Defaults to False.

        Raises:
            e: file not found error
            RuntimeError: raises error is no policy table has been computed and training is still required

        Returns:
            str: the decided action 
        """
        
        #map to change ints to strs 
        action_int_to_str = {0:'stick', 1:'hit'}
        
        # for training only 
        if training: 
            #get q value for hit and stick
            #try:
            stick_q = self.q_table_infinite[self.score-2][self.unused_ace][0]
            hit_q = self.q_table_infinite[self.score-2][self.unused_ace][1]
            # except IndexError as e: 
            #     print(e)
            #     print(f'score: {self.score} aces {self.unused_ace}')
            #     print(*self.hand)
            #     raise e
            
            #check if they are equal
            if stick_q == hit_q:
                #choose random action is yes
                action = np.random.randint(0,2)
            else:
                #else run epsilon greedy to find action 
                if self.epislon > np.random.random():
                    action = np.argmin(self.q_table_infinite[self.score-2][self.unused_ace])
                else:
                    action = np.argmax(self.q_table_infinite[self.score-2][self.unused_ace])
            
            return action_int_to_str[action]
        
        else: 
            #for no training reference policy table 
            try:
                #load policy table if not done
                if type(self.policy) != np.ndarray: 
                    self.policy = np.load('infinite_policy.npy')
            except FileNotFoundError as e:
                #if no policy can be found throw error
                print(e)
                raise RuntimeError('Training required to create policy table.')
            
            return action_int_to_str[self.policy[self.score-2][self.unused_ace]]
        
    def save_tables(self) -> None:
        """saves policy and q-tables and the end of training
        """
        np.save('infinite_q_table.npy', self.q_table_infinite)
        self.policy = np.zeros([19,2])

        for s_index, state in enumerate(self.q_table_infinite): 
            for u_index, unused_ace in enumerate(state):
                self.policy[s_index][u_index] = np.argmax(unused_ace)
                
        np.save('infinite_policy.npy', self.policy)
        
class Finite_agent(Agent):
    """An agent to play the finite version of the game

    Args:
        Agent (Agent): the parent class 
    """
    
    def __init__(self, hands):
        super().__init__(hands)
        self.q_table_finite = np.zeros([19,10,2,2])
        self.policy = None
        self.card_tracker = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
        self.loss_state = 0
        self.loss_state_tracker = np.zeros(10)
               
    def calculate_probability_of_loss(self):
        
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
        
    def assess(self, training):
        
        action_int_to_str = {0:'stick', 1:'hit'}
        # for training only 
        self.calculate_probability_of_loss()
        
        if training: 
            #get q value for hit and stick
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
        
        #! update win case 
        if new_card == None: 
            new_card_value = 0
        else:
            new_card_value = new_card.value
        
        print(self.loss_state)
        old_state = self.score
        new_state = old_state+new_card_value
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
        np.save('finite_q_table.npy', self.q_table_finite)
        self.policy = np.zeros([19,10,2])
        #! think of some better names here 
        for s_index, state in enumerate(self.q_table_finite): 
            for p_index, percentage in enumerate(state): 
                for u_index, unused_ace in enumerate(percentage):
                    self.policy[s_index][p_index][u_index] = np.argmax(unused_ace)
                
        np.save('finite_policy.npy', self.policy)
              
class Dealer(): 
    """A class for the dealer of the blackjack game 
    this is a passive dealer. 
    The dealer runs the game.
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
        #deck class
        deck = Deck()
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
            raise Exception('Interger above 0 needed.')
        
        
        
    def hit(self, is_infinite = False) -> Card:
        """Gives the player a random card when requested. 
        Either deletes the card from the deck when finite cards required. 
        Or keeps the card in the deck when is_infinite cards required.

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
        print(f'game ends with score {self.player.score} and reward {self.player.cumulative_reward}, hands {self.player.playable_hands}, cards {len(self.cards)}')
        if self.is_infinite == False: 
            for x,i in enumerate(self.player.loss_state_tracker):
                print(f'{x} - {i}')
        
            
dealer = Dealer(hands = 50000, is_infinite=True, training=True)
dealer.get_decks(1000)
dealer.play_game()

    
    
        

        