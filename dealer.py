from card import Deck, Card
import numpy as np

class Agent():
    """A class for the agent that plays blackjack 
    will contain q learning 
    """
    
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
    """A class for the dealer of the blackjack game 
    this is a passive dealer. 
    The dealer runs the game.
    """
    
    def __init__(self, episodes : int):
        
        self.cards = None
        self.player = Agent(episodes)
        
          
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
        
    def hit(self, infinate_cards = False) -> Card:
        """Gives the player a random card when requested. 
        Either deletes the card from the deck when finite cards required. 
        Or keeps the card in the deck when infinate cards required.

        Args:
            infinate_cards (bool, optional): argument to decide between infinate and finate . Defaults to False.

        Returns:
            Card: the selected card 
        """
        
        card_index = np.random.randint(0, len(self.cards))
        card = self.cards[card_index]
        
        if infinate_cards == False: 
            self.cards = np.delete(self.cards, card_index)
            
        return card
    
    def play_game(self):
        """Loops through the game until the number of cards runs out or the select
        number of episodes are finiished.
        """
        
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
                
                if len(self.cards) < 1:
                    break 
        
        print(f'game ends with score {self.player.score} and reward {self.player.cumulative_reward}, episodes {self.player.episodes}, cards {len(self.cards)}')
        
            
dealer = Dealer(500)
dealer.get_decks(1)
dealer.play_game()

    
    
        

        