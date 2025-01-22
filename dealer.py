from card import Deck, Card
import numpy as np

class Agent():
    
    def __init__(self, episodes : int):
        self.episodes = episodes
        self.hand = []
        self.score = 0
        self.cumulative_reward = 0

        
    def assess(_):
        print(inte:=np.random.randint(0,2))
        if inte:
            return 'hit'
        else: 
            return 'stick'
        
    def hit(self, new_card : Card) -> None:
        self.score += new_card.value
        self.hand.append(new_card)
        
    def reset_hand(self) -> None:
        self.hand = []
        self.score = 0

class Dealer(): 
    
    def __init__(self, episodes : int):
        
        self.cards = None
        self.player = Agent(episodes)
        
          
    def get_decks(self, num_deck : int) -> None: 
        
        deck = Deck()
        self.cards = np.array(deck.get_cards())
        
        if num_deck == 1: 
            
            return
        elif num_deck > 1:
            for _ in range(1,num_deck):
                self.cards = np.concatenate((self.cards, np.array(deck.get_cards())))
        else: 
            raise Exception('Interger above 0 needed.')
        
    def hit(self) -> Card:
        
        card_index = np.random.randint(0, len(self.cards))
        card = self.cards[card_index]
        np.delete(self.cards, card_index)
        return card
    
    def play_game(self):
        
        while(0 < self.player.episodes) and (0 < len(self.cards)):
            # ask player what they want to do,
            while True: 
                
                response = self.player.assess()
                
                if response == 'hit':
                    self.player.hit(self.hit())
                    print('player hits')
                elif response == 'stick':
                    #assess and give reward
                    self.player.cumulative_reward += self.player.score**2
                    print(f'player sticks with score {self.player.score} and reward {self.player.cumulative_reward}')
                    self.player.score = 0 
                    break
                
                if self.player.score > 21: 
                    print('player looses')
                    self.player.score = 0 
                    break 
            
                self.player.episodes -= 1
        
        print(f'game ends with score {self.player.score} and reward {self.player.cumulative_reward}')
        
            
dealer = Dealer(1)
dealer.get_decks(1) 
dealer.play_game()

    
    
        

        