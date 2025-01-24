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
        self.playable_hands = playable_hands
        self.unused_ace = 0
        self.hand = []
        self.score = 0
        self.cumulative_reward = 0


        #algorithm hyerparameters
        self.alpha = 0.1
        self.epislon = 0.1
        self.gamma = 0.9
        self.total_iter = playable_hands
    
    def update_q_table(self):
        raise NotImplementedError('update_q_table not implemented')
        
    def check_for_unused_ace(self):
        
        aces = [card for card in self.hand if (card.type == 'Ace') and (card.value == 11)]
        if len(aces) > 0:
            self.unused_ace = 1
                
    def change_ace_value(self):
        """finds the first unused ace in hand and changes it value down to 1
        """
        for card in self.hand:
            if (card.type == 'Ace') and (card.value == 11):
                card.change_value(1)
                self.score -= 10
                self.unused_ace = 0
        print('changed ace value')
        
    def hit(self, new_card : Card, training=False) -> None:
        new_score = new_card.value + self.score
        if training and (new_score < 22): 
            if new_score == 21: 
                print('yes?')
                self.update_q_table(new_card = new_card, action = 1, win_case = True)
            else: 
                self.update_q_table(new_card, 1)
        elif training and (new_card.type == 'Ace'):
            #use case for if ace recieved at 20
            
            #decrement value of ace 
            new_card.change_value(1)
            self.update_q_table(new_card = new_card, action = 1, win_case = True)
            #do win case q table update 
            
            
            
        self.score += new_card.value
        self.hand.append(new_card)
        self.check_for_unused_ace()
        
    def reset_hand(self) -> None:
        self.hand = []
        self.score = 0
        self.unused_ace = 0
        
    def save_tables(self):
        raise NotImplementedError('update_q_table not implemented')
        
class Infinite_agent(Agent):
    
    def __init__(self, hands):
        super().__init__(hands)
        self.q_table_infinite = np.zeros([19,2,2])
        self.policy = None
        
    def update_q_table(self, new_card : Card, action : int, win_case = False):
        
        #! update win case 
        if new_card == None: 
            new_card_value = 0
        else:
            new_card_value = new_card.value
        
        old_state = self.score
        new_state = old_state+new_card_value
        old_state_value = self.q_table_infinite[old_state-2][self.unused_ace][action] 
        
        reward = new_state**2 if action else self.score**2 
        
        if win_case or (action == 0): 
            max_future_value = 0
        elif action == 1:
            max_future_value = np.amax(self.q_table_infinite[new_state-2][self.unused_ace][action])
        
        self.alpha = 0.1/(2*math.exp(self.playable_hands/self.total_iter))
        
        #bellman eqaution 
        self.q_table_infinite[old_state-2][self.unused_ace][action] = \
            old_state_value + self.alpha*(reward + self.gamma*max_future_value + old_state_value)
            
        print('q-table updated')
        
    def assess(self, training):
        
        action_int_to_str = {0:'stick', 1:'hit'}
        # for training only 
        if training: 
            #get q value for hit and stick
            try:
                stick_q = self.q_table_infinite[self.score-2][self.unused_ace][0]
                hit_q = self.q_table_infinite[self.score-2][self.unused_ace][1]
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
                    action = np.argmin(self.q_table_infinite[self.score-2][self.unused_ace])
                else:
                    action = np.argmax(self.q_table_infinite[self.score-2][self.unused_ace])
            
            return action_int_to_str[action]
        
        else: 
            #! reference completed policy
            raise NotImplementedError('non-training version not implemented')
        
    def save_tables(self):
        np.save('infinite_q_table.npy', self.q_table_infinite)
        self.policy = np.zeros([19,2])
        #! think of some better names here 
        for s_index, state in enumerate(self.q_table_infinite): 
            for u_index, unused_ace in enumerate(state):
                print(s_index, u_index)
                self.policy[s_index][u_index] = np.argmax(unused_ace)
                
        np.save('infinite_policy.npy', self.policy)
        
class Dealer(): 
    """A class for the dealer of the blackjack game 
    this is a passive dealer. 
    The dealer runs the game.
    """
    
    def __init__(self, hands : int, is_infinite = False, training = False):
        
        self.cards = None
        self.is_infinite = is_infinite
        self.training = training
        
        if self.is_infinite:
            self.player = Infinite_agent(hands)
        else: 
            self.player = None
        
        
          
    def get_decks(self, num_deck : int) -> None: 
        """Collects the allotted number of decks 

        Args:
            num_deck (int): the required number of decks 

        Raises:
            Exception: if a value below 1 is entered an exception is thrown
        """
        deck = Deck()
        self.cards = np.array(deck.get_cards())
        
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
        
        card_index = np.random.randint(0, len(self.cards))
        card = copy.copy(self.cards[card_index])
        
        if is_infinite == False: 
            self.cards = np.delete(self.cards, card_index)
            
        return card
    
    def evaulate_stop_condition(self, is_infinite = False, decrement_hand = False) -> int:
        
        if is_infinite: 
            if decrement_hand:
                self.player.playable_hands -= 1 
            
            stop_condition = self.player.playable_hands
        else: 
            stop_condition = len(self.cards)
            
        return stop_condition
        
        
    def play_game(self):
        
        """Loops through the game until the number of cards runs out or the select
        number of hands are finiished.
        """
        stop_condition = self.evaulate_stop_condition(is_infinite=self.is_infinite)
        print(stop_condition)

        while(0 < stop_condition):
            
            # give player a card
            first_card = self.hit(is_infinite=self.is_infinite)
            self.player.score = first_card.value 
            self.player.hand.append(first_card)
            self.player.check_for_unused_ace()

            print('\nfirst hand given ----------------------------------------------')
            while True: 
                print(f'round begins with: score {self.player.score}, aces {self.player.unused_ace}')
                print(*self.player.hand)
            
                if self.player.score == 21: 
                    break
                
                if self.player.score > 21: 
                    if self.player.unused_ace == 0:
                        print('player looses')
                        break 
                    else:
                        self.player.change_ace_value()
                        print('got to over 21, changed ace value')
                
                response = self.player.assess(training=self.training)
                print(response)
                if response == 'hit':
                    self.player.hit(self.hit(is_infinite=self.is_infinite), training=self.training)
                    print('player hits')
                elif response == 'stick':
                    #assess and give reward
                    if self.training:
                        print('update from stick')
                        self.player.update_q_table(new_card = None, action = 0)
                    print(f'player sticks with score {self.player.score} and reward {self.player.cumulative_reward}')
                    break
                
                
                    
                if len(self.cards) < 1:
                    break 
                
            print(f'score {self.player.score}, hands {self.player.playable_hands}')
            
            self.player.reset_hand()
            stop_condition = self.evaulate_stop_condition(is_infinite=self.is_infinite, decrement_hand=self.is_infinite)
            
        if self.training: 
            print('epdisode of training complete')
            self.player.save_tables()      
            
        print(f'game ends with score {self.player.score} and reward {self.player.cumulative_reward}, hands {self.player.playable_hands}, cards {len(self.cards)}')
        
            
dealer = Dealer(hands = 10000, is_infinite=True, training=True)
dealer.get_decks(1)
dealer.play_game()

    
    
        

        