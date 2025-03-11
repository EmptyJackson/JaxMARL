"""Utility functions for the Monopoly Deal game."""
import jax.numpy as jnp
from .constants import *
from .state import EnvState

def get_color_name(color: int) -> str:
    """Get the string name of a color."""
    color_names = {
        BROWN: "Brown",
        LIGHT_BLUE: "Light Blue",
        PINK: "Pink",
        ORANGE: "Orange",
        RED: "Red",
        YELLOW: "Yellow",
        GREEN: "Green",
        DARK_BLUE: "Dark Blue",
        RAILROAD: "Railroad",
        UTILITY: "Utility",
    }
    return color_names.get(color, "Unknown")

def get_card_description(card: jnp.ndarray) -> str:
    """Get a human-readable description of a card."""
    card_type, value, color = card

    if card_type == MONEY_CARD:
        return f"{value}M"
    elif card_type == PROPERTY_CARD:
        if color >> 16:  # Wildcard
            color1 = color >> 16
            color2 = color & 0xFFFF
            return f"{get_color_name(color1)}/{get_color_name(color2)} Property"
        return f"{get_color_name(color)} Property"
    elif card_type == ACTION_CARD:
        action_names = {
            DEAL_BREAKER: "Deal Breaker",
            JUST_SAY_NO: "Just Say No",
            PASS_GO: "Pass Go",
            FORCED_DEAL: "Forced Deal",
            SLY_DEAL: "Sly Deal",
            DEBT_COLLECTOR: "Debt Collector",
            BIRTHDAY: "Birthday",
            DOUBLE_RENT: "Double Rent",
            HOUSE: "House",
            HOTEL: "Hotel",
        }
        return action_names.get(color, "Unknown Action")
    elif card_type == RENT_CARD:
        if color == -1:
            return "Wild Rent"
        colors = []
        for c in range(10):  # Check each color bit
            if color & (1 << c):
                colors.append(get_color_name(c))
        return f"Rent ({'/'.join(colors)})"
    return "Unknown Card"

def get_property_value(color: int) -> int:
    """Get the base monetary value of a property."""
    values = {
        BROWN: 1,
        LIGHT_BLUE: 1,
        PINK: 2,
        ORANGE: 2,
        RED: 3,
        YELLOW: 3,
        GREEN: 4,
        DARK_BLUE: 4,
        RAILROAD: 2,
        UTILITY: 2,
    }
    return values.get(color, 0)

def get_action_value(action_type: int) -> int:
    """Get the monetary value of an action card."""
    values = {
        DEAL_BREAKER: 5,
        JUST_SAY_NO: 4,
        PASS_GO: 1,
        FORCED_DEAL: 3,
        SLY_DEAL: 3,
        DEBT_COLLECTOR: 3,
        BIRTHDAY: 2,
        DOUBLE_RENT: 1,
        HOUSE: 3,
        HOTEL: 4,
    }
    return values.get(action_type, 0)

def get_required_set_size(color: int) -> int:
    """Get the number of cards required for a complete property set."""
    sizes = {
        BROWN: 2,
        LIGHT_BLUE: 3,
        PINK: 3,
        ORANGE: 3,
        RED: 3,
        YELLOW: 3,
        GREEN: 3,
        DARK_BLUE: 2,
        RAILROAD: 4,
        UTILITY: 2,
    }
    return sizes.get(color, 0)

def get_rent_value(properties: jnp.ndarray, color: int) -> int:
    """Calculate rent value for a color set including houses/hotels."""
    # Base rent values for complete sets
    base_values = {
        BROWN: 1,
        LIGHT_BLUE: 1,
        PINK: 2,
        ORANGE: 2,
        RED: 3,
        YELLOW: 3,
        GREEN: 4,
        DARK_BLUE: 4,
        RAILROAD: 2,
        UTILITY: 2,
    }

    # Count properties of the target color
    color_count = jnp.sum(properties[:, 2] == color)

    # Add wildcards that could be this color
    wild_count = jnp.sum((properties[:, 2] >> 16) == color)
    total_count = color_count + wild_count

    # Check if we have a complete set
    required = get_required_set_size(color)
    if total_count < required:
        return 0

    base_rent = base_values.get(color, 0)

    # Add house and hotel values if present
    house_present = jnp.any((properties[:, 0] == ACTION_CARD) & (properties[:, 2] == HOUSE))
    hotel_present = jnp.any((properties[:, 0] == ACTION_CARD) & (properties[:, 2] == HOTEL))

    total_rent = base_rent
    if house_present:
        total_rent += 3
    if hotel_present:
        total_rent += 4

    return total_rent

def has_complete_set(properties: jnp.ndarray, return_count=False) -> bool:
    """Check if a player has a complete set of properties of any color.

    Args:
        properties: Array of property cards
        return_count: If True, returns the count of complete sets instead of a boolean

    Returns:
        Boolean indicating if player has at least one complete set, or count of complete sets if return_count=True
    """
    complete_sets = 0
    for color in range(10):  # Loop through all colors
        count = int(jnp.sum(jnp.logical_and(properties[:, 0] == PROPERTY_CARD, properties[:, 2] == color)))
        required = get_required_set_size(color)
        if count >= required and required > 0:
            complete_sets += 1
            if not return_count:
                return True

    return complete_sets if return_count else False

def move_money(state: EnvState, from_player: int, amount: int, to_player: int) -> EnvState:
    """Move money cards from one player to another."""
    donor_bank = state.player_banks[from_player]

    # Find a card with exact value match
    indices = jnp.where((donor_bank[:, 1] == amount) & (donor_bank[:, 0] == MONEY_CARD))[0]
    if indices.shape[0] > 0:
        donor_index = int(indices[0])
        card_to_transfer = donor_bank[donor_index]
        new_donor_bank = donor_bank.at[donor_index].set(jnp.array([0, 0, 0], dtype=donor_bank.dtype))
    else:
        # If no exact match, try to find any money card
        indices = jnp.where(donor_bank[:, 0] == MONEY_CARD)[0]
        if indices.shape[0] == 0:
            return state  # No money cards to transfer

        donor_index = int(indices[0])
        card_to_transfer = donor_bank[donor_index]
        new_donor_bank = donor_bank.at[donor_index].set(jnp.array([0, 0, 0], dtype=donor_bank.dtype))

    # Find empty slot in recipient's bank
    recipient_bank = state.player_banks[to_player]
    empty_indices = jnp.where(jnp.all(recipient_bank == 0, axis=1))[0]
    if empty_indices.shape[0] == 0:
        return state  # No room in recipient's bank

    recipient_index = int(empty_indices[0])
    new_recipient_bank = recipient_bank.at[recipient_index].set(jnp.array(card_to_transfer, dtype=recipient_bank.dtype))

    # Update state with new banks
    new_player_banks = state.player_banks
    new_player_banks = new_player_banks.at[from_player].set(new_donor_bank)
    new_player_banks = new_player_banks.at[to_player].set(new_recipient_bank)

    return state.replace(player_banks=new_player_banks)