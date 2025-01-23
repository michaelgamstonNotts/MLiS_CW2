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
    
