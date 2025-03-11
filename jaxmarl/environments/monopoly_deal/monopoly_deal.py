"""
Monopoly Deal - A JAX Implementation
==================================

This module implements the Monopoly Deal card game in JAX, following the official rules.
The implementation is designed to be compatible with JAX's functional programming style
and supports vectorization for efficient parallel execution.

Game Overview:
-------------
- 2-5 players compete to collect 3 complete property sets
- Each turn consists of:
  1. Drawing 2 cards
  2. Playing up to 3 cards as actions
  3. Discarding down to 7 cards if necessary

Card Types:
-----------
1. Money Cards (20 total):
   - Values: 1M (6), 2M (5), 3M (3), 4M (3), 5M (2), 10M (1)
   - Used for paying rent and as currency

2. Property Cards (28 regular + 11 wildcards):
   - Regular properties in different colors
   - Wildcards can substitute for specific colors
   - Each color requires different set sizes:
     * Brown, Dark Blue: 2 cards
     * Most colors: 3 cards
     * Railroad: 4 cards
     * Utility: 2 cards

3. Action Cards (34 total):
   - Deal Breaker: Steal a complete set
   - Just Say No: Cancel an action
   - Pass Go: Draw 2 cards
   - Forced Deal: Force a property trade
   - Sly Deal: Steal a property
   - Debt Collector: Collect 5M from one player
   - Birthday: Collect 2M from each player
   - Double Rent: Double rent payment
   - House/Hotel: Add to property values

4. Rent Cards (13 total):
   - Color-specific rent cards
   - Wild rent cards
   - Rent value based on set size

Implementation Details:
---------------------
Card Encoding:
- card_type: Identifies basic card type (money, property, action, rent)
- value: Monetary value of the card (for bank/payments)
- color: Multi-purpose field encoding:
  * Properties: The color identifier
  * Actions: The action type
  * Rent: Target colors (using bitwise OR)
  * Wildcards: Two colors (color1 << 16 | color2)
  * For money: -1 (no color)

State Management:
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
import flax
import chex
from gymnasium import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

# For visualization
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PIL import Image

# Import from modular files
from .constants import *
from .state import EnvState, Card, PlayerState
from .utils import (
    get_color_name,
    get_card_description,
    get_property_value,
    get_action_value,
    get_required_set_size,
    get_rent_value,
    has_complete_set
)
from .actions import (
    handle_pass_go,
    handle_just_say_no,
    handle_house,
    handle_hotel,
    handle_double_rent,
    handle_rent,
    handle_sly_deal,
    handle_forced_deal,
    handle_debt_collector,
    handle_birthday,
    handle_deal_breaker,
    resolve_deal_breaker
)

class MonopolyDeal(MultiAgentEnv):
    """JAX-compatible implementation of Monopoly Deal card game.

    Key Features:
    - Fully vectorizable using JAX
    - Immutable state updates
    - Hidden information handling
    - Complete rule implementation

    The environment follows the standard OpenAI Gym interface with:
    - reset(): Initialize new game
    - step(): Process one action
    - render(): Visualize current state
    """

    # Card type constants
    MONEY_CARD = 0
    PROPERTY_CARD = 1
    ACTION_CARD = 2
    RENT_CARD = 3
    PASS_ACTION = -1

    # Property color constants
    BROWN = 0
    LIGHT_BLUE = 1
    PINK = 2
    ORANGE = 3
    RED = 4
    YELLOW = 5
    GREEN = 6
    DARK_BLUE = 7
    RAILROAD = 8
    UTILITY = 9

    # Action card type constants
    DEAL_BREAKER = 0
    JUST_SAY_NO = 1
    PASS_GO = 2
    FORCED_DEAL = 3
    SLY_DEAL = 4
    DEBT_COLLECTOR = 5
    BIRTHDAY = 6
    DOUBLE_RENT = 7
    HOUSE = 8
    HOTEL = 9

    # Card counts and values from rules
    CARD_COUNTS = {
        # Money cards (20 total)
        (MONEY_CARD, 1): 6,  # 1M cards
        (MONEY_CARD, 2): 5,  # 2M cards
        (MONEY_CARD, 3): 3,  # 3M cards
        (MONEY_CARD, 4): 3,  # 4M cards
        (MONEY_CARD, 5): 2,  # 5M cards
        (MONEY_CARD, 10): 1, # 10M card

        # Action cards (34 total)
        (ACTION_CARD, DEAL_BREAKER): 2,
        (ACTION_CARD, JUST_SAY_NO): 3,
        (ACTION_CARD, PASS_GO): 10,
        (ACTION_CARD, FORCED_DEAL): 3,
        (ACTION_CARD, SLY_DEAL): 3,
        (ACTION_CARD, DEBT_COLLECTOR): 3,
        (ACTION_CARD, BIRTHDAY): 3,
        (ACTION_CARD, DOUBLE_RENT): 2,
        (ACTION_CARD, HOUSE): 3,
        (ACTION_CARD, HOTEL): 2,

        # Property cards (28 total)
        (PROPERTY_CARD, BROWN): 2,
        (PROPERTY_CARD, LIGHT_BLUE): 3,
        (PROPERTY_CARD, PINK): 3,
        (PROPERTY_CARD, ORANGE): 3,
        (PROPERTY_CARD, RED): 3,
        (PROPERTY_CARD, YELLOW): 3,
        (PROPERTY_CARD, GREEN): 3,
        (PROPERTY_CARD, DARK_BLUE): 2,
        (PROPERTY_CARD, RAILROAD): 4,
        (PROPERTY_CARD, UTILITY): 2,

        # Rent cards (13 total)
        (RENT_CARD, DARK_BLUE | GREEN): 2,
        (RENT_CARD, RED | YELLOW): 2,
        (RENT_CARD, PINK | ORANGE): 2,
        (RENT_CARD, LIGHT_BLUE | BROWN): 2,
        (RENT_CARD, RAILROAD | UTILITY): 2,
        (RENT_CARD, -1): 3,  # Wild rent (can be used for any color)
    }

    # Property wildcards (11 total)
    PROPERTY_WILDCARDS = [
        (DARK_BLUE, GREEN),
        (GREEN, RAILROAD),
        (UTILITY, RAILROAD),
        (LIGHT_BLUE, RAILROAD),
        (LIGHT_BLUE, BROWN),
        (PINK, ORANGE),
        (PINK, ORANGE),
        (RED, YELLOW),
        (RED, YELLOW),
        (-1, -1),  # Multi-color wildcard
        (-1, -1),  # Multi-color wildcard
    ]

    # Forward declarations to satisfy linter:
    _enforce_hand_limit: Any
    _card_color: Any

    def __init__(
        self,
        num_players: int = 2,
        # Additional configuration parameters
    ):
        super().__init__(num_agents=num_players)
        assert 2 <= num_players <= 5, "Game supports 2-5 players (use two decks for 6+ players)"
        self.num_players = num_players
        self.agents = [str(i) for i in range(num_players)]  # Initialize agents as strings from 0 to num_players-1
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # Game configuration
        self.max_hand_size = 7
        self.initial_cards = 5
        self.cards_per_turn = 2
        self.max_actions = 3
        self.sets_to_win = 3
        # Initialize game parameters

    @property
    def name(self) -> str:
        return "MonopolyDeal-v0"

    def action_space(self, agent_id: str = None) -> spaces.Space:
        """Define the action space.

        Actions are encoded as follows:
        - -1: Pass turn
        - 0 to N-1: Play card from hand (where N is max hand size)
        - N to N+P-1: Select target player (where P is num_players) for action cards
        - N+P to N+P+C-1: Select target card (where C is max cards in play)
        """
        # Maximum number of possible actions:
        # - Pass turn (1)
        # - Play any card from hand (max_hand_size)
        # - Select any player (num_players)
        # - Select any card in play (max cards in play per player * num_players)
        max_cards_in_play = self.max_hand_size * self.num_players
        total_actions = 1 + self.max_hand_size + self.num_players + max_cards_in_play

        return spaces.Discrete(total_actions)

    def observation_space(self, agent_id: str = None) -> spaces.Space:
        """Define the observation space.

        Observation includes:
        - Player's hand (max_hand_size cards)
        - Each player's visible cards (properties and bank)
        - Current game state (turn, action count)
        - Card counts (deck size, discard size)

        Each card is encoded as a 3-tuple: (card_type, value, color)
        """
        # Space for a single card
        card_space = spaces.MultiDiscrete([
            4,      # card_type (money, property, action, rent)
            11,     # value (0-10)
            1<<17,  # color (including encoded wildcards)
        ])

        # Space for player's hand
        hand_space = spaces.Tuple([card_space] * self.max_hand_size)

        # Space for each player's visible cards
        player_visible_space = spaces.Dict({
            'properties': spaces.Tuple([card_space] * self.max_hand_size),
            'bank': spaces.Tuple([card_space] * self.max_hand_size)
        })

        # Complete observation space
        return spaces.Dict({
            'hand': hand_space,
            'players': spaces.Tuple([player_visible_space] * self.num_players),
            'game_state': spaces.MultiDiscrete([
                self.num_players,  # current_player
                self.max_actions + 1,  # action_count
                110,  # deck_size
                110   # discard_size
            ])
        })

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, EnvState]:
        """Reset the environment to initial state."""
        # Split keys for different random operations
        keys = jax.random.split(key, num=2)
        key_deck, key_first = keys

        # Initialize and shuffle deck
        deck = self._initialize_deck(key_deck)

        # Initialize empty arrays for player states with max hand size
        player_hands = jnp.zeros((self.num_players, self.max_hand_size, 3), dtype=jnp.int32)
        player_banks = jnp.zeros((self.num_players, self.max_hand_size, 3), dtype=jnp.int32)
        player_properties = jnp.zeros((self.num_players, self.max_hand_size, 3), dtype=jnp.int32)

        # Create initial state
        state = EnvState(
            deck=deck,
            discard=jnp.zeros((0, 3), dtype=jnp.int32),
            player_hands=player_hands,
            player_banks=player_banks,
            player_properties=player_properties,
            current_player=jax.random.randint(key_first, (), 0, self.num_players),
            action_count=0,
            done=False,
            double_rent_active=False,
            action_cancelled=False,
            pending_action=None,
            pending_action_target=None,
            pending_action_source=None
        )

        print("\nDealing initial cards:")
        # Deal initial cards to each player
        remaining_deck = deck
        new_player_hands = player_hands

        # Deal exactly 5 cards to each player
        for card_idx in range(self.initial_cards):
            for player_idx in range(self.num_players):
                # Draw top card and remove from deck
                drawn_card = remaining_deck[0]
                remaining_deck = remaining_deck[1:]

                # Validate card values
                card_type = jnp.clip(drawn_card[0], 0, 3)  # 0-3 are valid card types
                value = jnp.clip(drawn_card[1], 0, 10)  # 0-10 are valid values
                color = jnp.clip(drawn_card[2], -1, 9)  # -1 to 9 are valid colors
                validated_card = jnp.array([card_type, value, color], dtype=jnp.int32)

                print(f"  Player {player_idx} drawing card {card_idx+1}/5: {validated_card}")

                # Add card to the pre-allocated slot
                new_player_hands = new_player_hands.at[player_idx, card_idx].set(validated_card)

                # Print the player's current hand for verification
                valid_cards = new_player_hands[player_idx, :card_idx+1]
                print(f"  Player {player_idx} hand ({card_idx+1} cards): {valid_cards}")

        print(f"\nFinal deck size: {len(remaining_deck)}")
        print("Initial hands:")
        for player_idx in range(self.num_players):
            valid_cards = new_player_hands[player_idx, :self.initial_cards]
            print(f"Player {player_idx}: {valid_cards}")

        # Update state with dealt cards and remaining deck
        state = state.replace(
            deck=remaining_deck,
            player_hands=new_player_hands
        )

        # Get initial observations
        obs = self.get_obs(state)

        return obs, state

    def step(self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, int]) -> Tuple[Dict, EnvState, Dict, Dict, Dict]:
        """Execute one step in the environment."""
        if actions:
            acting_agent = list(actions.keys())[0]
            state = state.replace(current_player=int(acting_agent))

        # Process the action
        current_player = str(state.current_player)
        if current_player not in actions:
            # If current player's action is missing, use pass action
            action = self.PASS_ACTION
        else:
            action = actions[current_player]

        # If there is a pending action and the current player (target) chooses to PASS, resolve the pending Deal Breaker
        if state.pending_action is not None and action == self.PASS_ACTION and int(state.current_player) == state.pending_action_target:
            original = state.pending_action if state.pending_action_original is None else state.pending_action_original
            card_color = int(original[2])
            if card_color == self.DEAL_BREAKER:
                state = resolve_deal_breaker(state, original)
            elif card_color == self.SLY_DEAL:
                state = handle_sly_deal(state, original)
            elif card_color == self.FORCED_DEAL:
                state = handle_forced_deal(state, original)
            elif card_color == self.BIRTHDAY:
                state = handle_birthday(state, original)
            elif card_color == self.DOUBLE_RENT:
                state = handle_double_rent(state, original)
            elif card_color == self.HOUSE:
                state = handle_house(state, original)
            elif card_color == self.HOTEL:
                state = handle_hotel(state, original)
            state = state.replace(pending_action=None, pending_action_target=None, pending_action_source=None, pending_action_original=None, pending_counter=0)

        new_state = self._process_action(state, action) or state
        if new_state is None:
            new_state = state
        # If there's a pending action, move to target player's turn
        if new_state.pending_action is not None:
            new_state = new_state.replace(
                current_player=new_state.pending_action_target
            )
        # Otherwise, if we've used all actions or passed, move to next player
        elif action == self.PASS_ACTION or new_state.action_count >= 3:
            # Enforce hand limit for the current player's hand before ending turn
            active = int(state.current_player)
            new_state = self._enforce_hand_limit(new_state, active)

            new_state = new_state.replace(
                action_count=0,
                double_rent_active=False,
                pending_action=None,
                pending_action_target=None,
                pending_action_source=None
            )

            next_player = (active + 1) % len(self.agents)
            new_state = new_state.replace(current_player=next_player)

            # Draw 2 cards for next player
            keys = jax.random.split(key, 2)
            for i in range(2):
                new_state = self._draw_card(new_state, next_player, keys[i])

        # Otherwise, increment action count
        else:
            new_state = new_state.replace(
                action_count=new_state.action_count + 1
            )

        # Check win condition
        winner = -1
        for player_idx in range(len(self.agents)):
            if self._check_win_condition(new_state, player_idx):
                winner = player_idx
                break

        if winner != -1:
            # Set done for all players
            dones = {str(i): True for i in range(self.num_players)}
            # Set reward of 1 for winner, -1 for others
            rewards = {str(i): 1.0 if i == winner else -1.0 for i in range(self.num_players)}
        else:
            # Game continues
            dones = {str(i): False for i in range(self.num_players)}
            rewards = {str(i): 0.0 for i in range(self.num_players)}

        # Get observations
        obs = self.get_obs(new_state)

        # Return step information
        infos = {str(i): {} for i in range(self.num_players)}
        return obs, new_state, rewards, dones, infos

    def get_obs(self, state: EnvState) -> Dict:
        """Get observations for all players.

        Each player observes:
        - Their own hand (only the actual cards, not padding)
        - All players' visible cards (properties and bank)
        - Game state information
        """
        observations = {}

        for player_idx, agent in enumerate(self.agents):
            # Get player's hand (only visible to them)
            hand = state.player_hands[player_idx]

            # Only include actual cards (first initial_cards slots)
            actual_hand = hand[:self.initial_cards]

            # Get all players' visible cards
            players_visible = []
            for p_idx in range(self.num_players):
                # Get player's properties and bank
                properties = state.player_properties[p_idx]
                bank = state.player_banks[p_idx]

                # Create fixed-size arrays for properties and bank
                fixed_properties = jnp.zeros((self.max_hand_size, 3), dtype=jnp.int32)
                fixed_bank = jnp.zeros((self.max_hand_size, 3), dtype=jnp.int32)

                # Copy non-zero cards
                fixed_properties = fixed_properties.at[:len(properties)].set(properties)
                fixed_bank = fixed_bank.at[:len(bank)].set(bank)

                players_visible.append({
                    'properties': fixed_properties,
                    'bank': fixed_bank
                })

            # Create observation for this player
            observations[agent] = {
                'hand': actual_hand,
                'players': tuple(players_visible),
                'game_state': jnp.array([
                    state.current_player,
                    state.action_count,
                    len(state.deck),
                    len(state.discard)
                ], dtype=jnp.int32)
            }

        return observations

    def render(self, state: EnvState = None, mode="human"):
        """Render the current game state."""
        if state is None:
            return None

        fig = Figure(figsize=(15, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(-7, 7)
        ax.set_ylim(-5, 5)

        # Draw deck and discard pile
        deck_size = len(state.deck)
        discard_size = len(state.discard)

        # Draw deck area with blue background
        ax.add_patch(Rectangle((-6.5, -4), 1, 1.5, facecolor='lightblue', alpha=0.3))
        ax.text(-6, -3.2, f"Deck\n{deck_size} cards", ha='center', fontsize=10)

        # Draw discard area with pink background
        ax.add_patch(Rectangle((5.5, -4), 1, 1.5, facecolor='pink', alpha=0.3))
        ax.text(6, -3.2, f"Discard\n{discard_size} cards", ha='center', fontsize=10)

        # Draw player areas
        num_players = len(state.player_hands)
        for i in range(num_players):
            y = 3 - 2 * i
            # Highlight current player's area in light red
            color = 'mistyrose' if i == state.current_player else 'whitesmoke'
            ax.add_patch(Rectangle((-6, y-0.8), 12, 1.6, facecolor=color, alpha=0.3))

            # Player label with bold for current player
            weight = 'bold' if i == state.current_player else 'normal'
            ax.text(-5.8, y+0.5, f"Player {i}", fontsize=12, fontweight=weight)

            # Hand info
            hand = state.player_hands[i]
            hand_size = len(jnp.where(jnp.any(hand != 0, axis=1))[0])
            hand_text = f"Hand ({hand_size} cards):"
            if i == state.current_player:
                hand_text += "\n" + "\n".join(
                    f"  {self._get_card_description(card)}"
                    for card in hand
                    if jnp.any(card != 0)
                )
            ax.text(-5.8, y-0.2, hand_text, fontsize=9)

            # Bank info with money symbol
            bank = state.player_banks[i]
            bank_size = len(bank) if len(bank.shape) > 1 else 0
            bank_value = 0
            if bank_size > 0:
                bank_cards = bank[jnp.any(bank != 0, axis=1)]
                bank_value = jnp.sum(bank_cards[:, 1]) if len(bank_cards) > 0 else 0
            bank_text = f"$ Bank: ${bank_value}M ({bank_size} cards)"
            if bank_size > 0:
                bank_text += "\n" + "\n".join(
                    f"  {self._get_card_description(card)}"
                    for card in bank
                    if jnp.any(card != 0)
                )
            ax.text(-2, y-0.2, bank_text, fontsize=9)

            # Properties info with building symbol
            props = state.player_properties[i]
            props_size = len(props) if len(props.shape) > 1 else 0
            props_by_color = {}
            if props_size > 0:
                prop_cards = props[jnp.any(props != 0, axis=1)]
                for card in prop_cards:
                    if jnp.any(card != 0):
                        color = self._get_color_name(int(card[2]))
                        props_by_color[color] = props_by_color.get(color, 0) + 1

            prop_text = f"# Properties ({props_size} cards):\n"
            for color, count in props_by_color.items():
                required = 3  # Default set size
                if color in ["Brown", "Dark Blue", "Utility"]:
                    required = 2
                elif color == "Railroad":
                    required = 4
                # Add completion indicator
                complete = "✓" if count >= required else " "
                prop_text += f"  {complete} {color}: {count}/{required}\n"
            ax.text(2, y-0.2, prop_text, fontsize=9)

        # Game status with simple symbols
        status_text = f"* Current Player: {state.current_player}\n"
        status_text += f"* Actions Left: {3 - state.action_count}\n"
        if state.double_rent_active:
            status_text += "* Double Rent Active!\n"
        if state.pending_action is not None:
            status_text += f"* Pending Action from Player {state.pending_action_source}\n"
            status_text += f"* Target: Player {state.pending_action_target}\n"
        ax.text(0, 4, status_text, ha='center', fontsize=11)

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        ax.set_title("Monopoly Deal", fontsize=14, pad=20)

        if mode == "human":
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            buffer = canvas.buffer_rgba()
            image = np.asarray(buffer)
            return Image.fromarray(image)
        else:
            return fig

    def _get_color_name(self, color: int) -> str:
        """Convert color code to human-readable name."""
        if color == -1:
            return "None"
        color_names = {
            self.BROWN: "Brown",
            self.LIGHT_BLUE: "Light Blue",
            self.PINK: "Pink",
            self.ORANGE: "Orange",
            self.RED: "Red",
            self.YELLOW: "Yellow",
            self.GREEN: "Green",
            self.DARK_BLUE: "Dark Blue",
            self.RAILROAD: "Railroad",
            self.UTILITY: "Utility"
        }
        return color_names.get(color, f"Unknown({color})")

    def _get_card_description(self, card: chex.Array) -> str:
        """Generate human-readable card description."""
        card_type = int(card[0])
        value = int(card[1])
        color = int(card[2])

        if card_type == self.MONEY_CARD:
            return f"${value}M"
        elif card_type == self.PROPERTY_CARD:
            return f"{self._get_color_name(color)} Property"
        elif card_type == self.ACTION_CARD:
            action_names = {
                self.DEAL_BREAKER: "Deal Breaker",
                self.JUST_SAY_NO: "Just Say No",
                self.PASS_GO: "Pass Go",
                self.FORCED_DEAL: "Forced Deal",
                self.SLY_DEAL: "Sly Deal",
                self.DEBT_COLLECTOR: "Debt Collector",
                self.BIRTHDAY: "Birthday",
                self.DOUBLE_RENT: "Double Rent",
                self.HOUSE: "House",
                self.HOTEL: "Hotel"
            }
            return action_names.get(color, f"Unknown Action({color})")
        elif card_type == self.RENT_CARD:
            colors = []
            for c in range(10):
                if color & (1 << c):
                    colors.append(self._get_color_name(c))
            return f"Rent ({' & '.join(colors)})"
        return f"Unknown Card Type({card_type})"

    def create_animation(self, state_seq: List[EnvState], filename: str = "game.gif", duration: int = 500):
        """Create an animated GIF from a sequence of game states.

        Args:
            state_seq: List of EnvState objects representing the game progression
            filename: Output filename for the GIF
            duration: Duration for each frame in milliseconds
        """
        # Render each state as a frame
        frames = []
        for state in state_seq:
            img = self.render(state)
            frames.append(img)

        # Save as animated GIF
        if len(frames) > 0:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )

    # Helper methods
    def _initialize_deck(self, key: chex.PRNGKey) -> chex.Array:
        """Create and shuffle the initial deck.

        Cards are represented as [card_type, value, color] arrays.

        Args:
            key: JAX random key for shuffling

        Returns:
            chex.Array: Shuffled deck where each card is [card_type, value, color]
        """
        cards = []

        # Create money cards
        for (card_type, value), count in self.CARD_COUNTS.items():
            if card_type == self.MONEY_CARD:
                for _ in range(count):
                    cards.append([self.MONEY_CARD, value, -1])  # -1 for no color

        # Create property cards
        for (card_type, color), count in self.CARD_COUNTS.items():
            if card_type == self.PROPERTY_CARD:
                for _ in range(count):
                    cards.append([self.PROPERTY_CARD, self._get_property_value(color), color])

        # Create property wildcards
        for color1, color2 in self.PROPERTY_WILDCARDS:
            cards.append([self.PROPERTY_CARD, 0, (color1 << 16) | color2])

        # Create action cards
        for (card_type, action_type), count in self.CARD_COUNTS.items():
            if card_type == self.ACTION_CARD:
                for _ in range(count):
                    cards.append([self.ACTION_CARD, self._get_action_value(action_type), action_type])

        # Create rent cards
        for (card_type, colors), count in self.CARD_COUNTS.items():
            if card_type == self.RENT_CARD:
                for _ in range(count):
                    cards.append([self.RENT_CARD, 1, colors])  # Rent cards have 1M value

        # Convert to array and shuffle
        deck = jnp.array(cards, dtype=jnp.int32)
        shuffled_indices = jax.random.permutation(key, len(deck))
        return deck[shuffled_indices]

    def _get_property_value(self, color: int) -> int:
        """Get the monetary value of a property card based on its color"""
        property_values = {
            self.BROWN: 1,
            self.LIGHT_BLUE: 1,
            self.PINK: 2,
            self.ORANGE: 2,
            self.RED: 3,
            self.YELLOW: 3,
            self.GREEN: 4,
            self.DARK_BLUE: 4,
            self.RAILROAD: 2,
            self.UTILITY: 2
        }
        return property_values.get(color, 0)

    def _get_action_value(self, action_type: int) -> int:
        """Get the monetary value of an action card based on its type"""
        action_values = {
            self.DEAL_BREAKER: 5,
            self.JUST_SAY_NO: 4,
            self.PASS_GO: 1,
            self.FORCED_DEAL: 3,
            self.SLY_DEAL: 3,
            self.DEBT_COLLECTOR: 3,
            self.BIRTHDAY: 2,
            self.DOUBLE_RENT: 1,
            self.HOUSE: 3,
            self.HOTEL: 4
        }
        return action_values.get(action_type, 0)

    def _deal_cards(self, state: EnvState, key: chex.PRNGKey) -> EnvState:
        """Deal initial cards to all players.

        Each player starts with 5 cards.

        Args:
            state: Current game state
            key: JAX random key

        Returns:
            Updated game state with cards dealt to players
        """
        new_state = state
        cards_per_player = 5

        # Deal cards to each player
        for player_idx in range(len(self.agents)):
            # Draw 5 cards for this player
            for _ in range(cards_per_player):
                if len(new_state.deck) == 0:
                    # If deck is empty, shuffle discard pile into deck
                    if len(new_state.discard) == 0:
                        break  # No cards left to deal

                    key, subkey = jax.random.split(key)
                    shuffled_indices = jax.random.permutation(subkey, len(new_state.discard))
                    new_deck = new_state.discard[shuffled_indices]
                    new_state = new_state.replace(
                        deck=new_deck,
                        discard=jnp.array([])
                    )

                if len(new_state.deck) > 0:
                    # Draw top card
                    drawn_card = new_state.deck[0]
                    new_deck = new_state.deck[1:]

                    # Add to player's hand
                    player_hand = state.player_hands[player_idx]
                    new_hand = jnp.append(player_hand, drawn_card)
                    new_player_hands = state.player_hands.at[player_idx].set(new_hand)

                    # Update state
                    new_state = new_state.replace(
                        deck=new_deck,
                        player_hands=new_player_hands
                    )

        return new_state

    def _process_action(self, state: EnvState, action: int) -> EnvState:
        current_player = int(state.current_player)

        # If it's a pass action, return state as is
        if action == self.PASS_ACTION:
            return state

        # Get the card being played
        card = state.player_hands[current_player][action]
        card_type = int(card[0])
        card_color = int(card[2])

        # Handle different card types
        if card_type == self.ACTION_CARD:
            if card_color == self.SLY_DEAL:
                return handle_sly_deal(state, action)
            elif card_color == self.HOUSE:
                return handle_house(state, action)
            elif card_color == self.HOTEL:
                return handle_hotel(state, action)
            elif card_color == self.JUST_SAY_NO:
                return handle_just_say_no(state, action)
            elif card_color == self.PASS_GO:
                return handle_pass_go(state, action)
            elif card_color == self.DEBT_COLLECTOR:
                return handle_debt_collector(state, action)
            elif card_color == self.BIRTHDAY:
                return handle_birthday(state, action)
            elif card_color == self.DOUBLE_RENT:
                return handle_double_rent(state, action)
            elif card_color == self.DEAL_BREAKER:
                # Remove deal breaker card from hand
                new_hand = jnp.delete(state.player_hands[current_player], action, axis=0)
                # Pad with zeros to maintain original shape
                new_hand = jnp.pad(new_hand, ((0, 1), (0, 0)), constant_values=0)
                # Add the deal breaker card to discard pile
                new_discard = jnp.append(state.discard, card.reshape(1, -1), axis=0)

                # Find target with complete set
                target = -1
                for i in range(len(self.agents)):
                    if i == current_player:
                        continue
                    if has_complete_set(state.player_properties[i]):
                        target = i
                        break

                new_state = state.replace(
                    player_hands=state.player_hands.at[current_player].set(new_hand),
                    discard=new_discard,
                    pending_action=card,
                    pending_action_source=current_player,
                    pending_action_target=target
                )
                return new_state
            elif card_color == self.FORCED_DEAL:
                return handle_forced_deal(state, action)
        elif card_type == self.MONEY_CARD:
            return self._move_card_to_bank(state, action)
        elif card_type == self.PROPERTY_CARD:
            return self._move_card_to_properties(state, action)
        elif card_type == self.RENT_CARD:
            return handle_rent(state, action)

        # If no handler matched, return state unchanged
        return state

    def _check_win_condition(self, state: EnvState, player_idx: int) -> bool:
        """Check if any player has won the game.

        A player wins by collecting three complete property sets of different colors.
        """
        # Use the imported has_complete_set function
        complete_sets = has_complete_set(state.player_properties[player_idx], return_count=True)
        return complete_sets >= self.sets_to_win

    def _draw_card(self, state: EnvState, player_idx: int, key: chex.PRNGKey) -> EnvState:
        """Draw a card from the deck and add it to player's hand."""
        # If deck is empty, shuffle discard pile into deck
        if len(state.deck) == 0:
            if len(state.discard) == 0:
                return state  # No cards left to draw

            key, subkey = jax.random.split(key)
            shuffled_indices = jax.random.permutation(subkey, len(state.discard))
            new_deck = state.discard[shuffled_indices]
            state = state.replace(
                deck=new_deck,
                discard=jnp.zeros((0, 3), dtype=jnp.int32)
            )

        if len(state.deck) > 0:
            # Draw top card and remove from deck
            drawn_card = state.deck[0]
            new_deck = state.deck[1:]

            # Count non-zero cards to find current hand size
            hand_size = jnp.sum(jnp.any(state.player_hands[player_idx] != 0, axis=1))

            # Only add card if we haven't reached max_hand_size
            if hand_size < self.max_hand_size:
                # Validate card values
                card_type = jnp.clip(drawn_card[0], 0, 3)  # 0-3 are valid card types
                value = jnp.clip(drawn_card[1], 0, 10)  # 0-10 are valid values
                color = jnp.clip(drawn_card[2], -1, 9)  # -1 to 9 are valid colors
                validated_card = jnp.array([card_type, value, color], dtype=jnp.int32)

                # Add card to first empty slot
                new_player_hands = state.player_hands.at[player_idx, hand_size].set(validated_card)

                # Update state
                state = state.replace(
                    deck=new_deck,
                    player_hands=new_player_hands
                )

        return state

    def _move_card_to_bank(self, state: EnvState, card_idx: int) -> EnvState:
        """Move a card from player's hand to their bank"""
        current_player = state.current_player
        player_hand = state.player_hands[current_player]

        # Get the card
        card = player_hand[card_idx]

        # Create a mask for all cards except the one being removed
        mask = jnp.arange(len(player_hand)) != card_idx
        mask_full = jnp.broadcast_to(mask[:, None], player_hand.shape)
        new_hand = jnp.where(mask_full, player_hand, 0)

        # Find first empty slot in bank (where all values are 0)
        bank = state.player_banks[current_player]
        is_empty = jnp.all(bank == 0, axis=1)
        empty_slot = jnp.argmax(is_empty)  # First True value
        new_bank = bank.at[empty_slot].set(card)

        # Update state
        return state.replace(
            player_hands=state.player_hands.at[current_player].set(new_hand),
            player_banks=state.player_banks.at[current_player].set(new_bank)
        )

    def _move_card_to_properties(self, state: EnvState, card_idx: int) -> EnvState:
        """Move a property card from player's hand to their property area"""
        current_player = state.current_player
        player_hand = state.player_hands[current_player]

        # Get the card
        card = player_hand[card_idx]

        # Create a mask for all cards except the one being removed
        mask = jnp.arange(len(player_hand)) != card_idx
        mask_full = jnp.broadcast_to(mask[:, None], player_hand.shape)
        new_hand = jnp.where(mask_full, player_hand, 0)

        # Find first empty slot in properties (where all values are 0)
        props = state.player_properties[current_player]
        is_empty = jnp.all(props == 0, axis=1)
        empty_slot = jnp.argmax(is_empty)  # First True value
        new_properties = props.at[empty_slot].set(card)

        # Update state
        return state.replace(
            player_hands=state.player_hands.at[current_player].set(new_hand),
            player_properties=state.player_properties.at[current_player].set(new_properties)
        )

    def _enforce_hand_limit(self, state: EnvState, player_idx: int) -> EnvState:
        """Enforce the 7-card hand limit rule.

        If a player has more than 7 cards in their hand at the end of their turn,
        they must discard down to 7 cards. The discarded cards go to the discard pile.

        Args:
            state: Current game state
            player_idx: Index of the player whose hand to check

        Returns:
            Updated game state with hand limit enforced
        """
        hand = state.player_hands[player_idx]
        hand_size = jnp.sum(jnp.any(hand != 0, axis=1))

        if hand_size > 7:
            # Move excess cards to discard pile
            excess_cards = hand[7:]
            new_hand = jnp.zeros_like(hand)
            new_hand = new_hand.at[:7].set(hand[:7])

            # Update discard pile with excess cards
            discard = state.discard
            discard = jnp.vstack([discard, excess_cards[excess_cards != 0].reshape(-1, 3)])

            # Update state
            state = state.replace(
                player_hands=state.player_hands.at[player_idx].set(new_hand),
                discard=discard
            )

        return state
