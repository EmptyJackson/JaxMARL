"""State classes for the Monopoly Deal game."""
from dataclasses import dataclass
from typing import Optional
import chex

@dataclass
class Card:
    """Documentation class representing a card's structure.
    In the actual implementation, cards are stored as [card_type, value, color] arrays
    for JAX compatibility.

    Attributes:
        card_type: Integer identifying card category (money, property, action, rent)
        value: Monetary value of the card (for bank/payments)
        color: Multi-purpose field encoding:
            - For properties: The color identifier
            - For actions: The action type
            - For rent: Target colors (using bitwise OR)
            - For wildcards: Two colors (color1 << 16 | color2)
            - For money: -1 (no color)
    """
    card_type: int
    value: int
    color: int

@dataclass
class PlayerState:
    """Represents a player's current game state.

    Cards are stored as [N, 3] arrays where N is the number of cards and
    each card is represented as [card_type, value, color].

    Attributes:
        hand: [N, 3] array of cards in player's hand (private information)
        bank: [N, 3] array of money cards played (public information)
        properties: [N, 3] array of property cards played (public information)
    """
    hand: chex.Array      # [N, 3] array of cards
    bank: chex.Array      # [N, 3] array of cards
    properties: chex.Array # [N, 3] array of cards

@dataclass
class EnvState:
    """Represents the complete game state.

    Attributes:
        deck: [N, 3] array of remaining cards in draw pile
        discard: [N, 3] array of cards in discard pile
        player_hands: [num_players, N, 3] array of cards in each player's hand
        player_banks: [num_players, N, 3] array of money cards played by each player
        player_properties: [num_players, N, 3] array of property cards played by each player
        current_player: Index of active player
        action_count: Number of actions taken this turn (max 3)
        done: Whether game has ended
        double_rent_active: Flag indicating if double rent is active
        action_cancelled: Flag indicating if current action was cancelled by Just Say No
        pending_action: Card being played that can be responded to with Just Say No
        pending_action_target: Player targeted by pending action
        pending_action_source: Player who played the pending action
        pending_action_original: The original pending action card (for Just Say No chaining)
        pending_counter: Counter for Just Say No chaining (0 or 1)
        pending_action_original_source: Optional[int] = None
        pending_action_original_target: Optional[int] = None
    """
    deck: chex.Array
    discard: chex.Array
    player_hands: chex.Array
    player_banks: chex.Array
    player_properties: chex.Array
    current_player: int
    action_count: int
    done: bool
    double_rent_active: bool
    action_cancelled: bool
    pending_action: Optional[chex.Array]
    pending_action_target: Optional[int]
    pending_action_source: Optional[int]
    pending_action_original: Optional[chex.Array] = None
    pending_counter: int = 0
    pending_action_original_source: Optional[int] = None
    pending_action_original_target: Optional[int] = None

    def replace(self, **updates):
        """Return a new state with updated values."""
        return EnvState(**{**self.__dict__, **updates})