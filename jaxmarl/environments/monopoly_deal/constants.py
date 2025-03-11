"""Constants for the Monopoly Deal game."""

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