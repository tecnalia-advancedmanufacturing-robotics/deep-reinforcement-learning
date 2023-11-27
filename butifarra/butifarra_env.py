import functools

import gymnasium
import numpy as np
import gymnasium.spaces as spaces
from termcolor import colored
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

POINTS = (0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5)
RANKS = ('2', '3', '4', '5', '6', '7', '8', '10', '11', '12', 'A', '9')
SUITS = ('O', 'C', 'E', 'B')
colors = ('yellow', 'red', 'blue', 'green')


def get_card(id):
    return (id % 12, id // 12)


def get_action(card: str):
    rank = RANKS.index(card[:-1])
    suit = SUITS.index(card[-1])
    return rank + suit * 12


def beats(id1, id2, trump, playing_suit):
    card1 = get_card(id1)
    card2 = get_card(id2)
    if card1[1] == trump and card2[1] != trump:
        return True
    if card1[1] != trump and card2[1] == trump:
        return False
    if card1[1] == trump and card2[1] == trump:
        return card1[0] > card2[0]
    if card1[1] == playing_suit and card2[1] != playing_suit:
        return True
    if card1[1] != playing_suit and card2[1] == playing_suit:
        return False
    return card1[0] > card2[0]


def get_winner(table, trump):
    playing_suit = get_card(table[0])[1]
    if len(table) == 0:
        return None
    winner = table[0]
    for card in table[1:]:
        if beats(card, winner, trump, playing_suit):
            winner = card
    return table.index(winner)


def format_card(card):
    if isinstance(card, list):
        return '[' + ', '.join([format_card(c) for c in card]) + ']'
    rank, suit = get_card(card)
    string = f"{RANKS[rank]}{SUITS[suit]}"
    return colored(string, colors[suit])


def make_one_hot(value, size=48):
    one_hot = np.zeros(size, dtype=np.int8)
    one_hot[value] = 1
    return one_hot


def make_one_hot_list(values, size=48, n=4):
    one_hot = np.zeros((n, size), dtype=np.int8)
    empty = n - len(values)
    for i, value in enumerate(values):
        one_hot[i + empty][value] = 1
    return one_hot.flatten()


def from_one_hot(one_hot):
    return np.where(one_hot)


def legal_moves(hand, table, trump):
    if len(table) == 0:
        return hand
    playing_suit = table[0] // 12
    legal_cards = [card for card in hand if card // 12 == playing_suit]
    if len(legal_cards) == 0:
        legal_cards = hand
    winner = get_winner(table, trump)
    need_to_win = winner != len(table) - 2
    if need_to_win:
        legal_winners = [card for card in legal_cards if beats(card, table[winner], trump, playing_suit)]
        if len(legal_winners) > 0:
            legal_cards = legal_winners
    return legal_cards


def ButifarraEnv(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None, flatten=True):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(4)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as
        # attributes to be used in their corresponding methods

        self.obs = spaces.Dict({
            # Which type is trump
            "trump": spaces.MultiBinary(4),
            # Which cards are in the hand
            "hand": spaces.MultiBinary(48),
            # Which cards are visible on the table
            "table": spaces.MultiBinary(48 * 4),
            # Which cards are the last trick won by the other team
            "last_trick_theirs": spaces.MultiBinary(48 * 4),
            # Which cards are the last trick won by our team
            "last_trick_ours": spaces.MultiBinary(48 * 4),
        })

        self.flatten = flatten

        obs = self.obs
        if self.flatten:
            obs = spaces.flatten_space(self.obs)

        self._action_spaces = {agent: spaces.Discrete(48) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": obs,
                "action_mask": spaces.MultiBinary(48)
            })
            for agent in self.possible_agents
        }
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self._observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for agent in self.agents:
            print(f"Agent {agent} has hand {format_card(self.hands[agent])}")
        print(f"Trump is {SUITS[self.trump]}")
        print(f"Table is {format_card(self.table)}")
        print(f"Last trick won by 02 team is {format_card(self.last_trick_02)}")
        print(f"Last trick won by 13 team is {format_card(self.last_trick_13)}")

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        observation = {
            # Which type is trump
            "trump": make_one_hot(self.trump, 4),
            # Which cards are in the hand
            "hand": make_one_hot(self.hands[agent]),
            # Which cards are visible on the table
            "table": make_one_hot_list(self.table),
            # Which cards are the last trick won by our team
            "last_trick_ours": make_one_hot_list(self.last_trick_02 if agent in ("player_0", "player_1") else self.last_trick_13),
            # Which cards are the last trick won by the other team
            "last_trick_theirs": make_one_hot_list(self.last_trick_02 if agent in ("player_0", "player_1") else self.last_trick_13),
        }

        if self.flatten:
            observation = spaces.flatten(self.obs, observation)

        res = {
            "observation": observation,
            # Which are the legal moves
            "action_mask": make_one_hot(legal_moves(self.hands[agent], self.table, self.trump)),
        }
        return res

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        deck = list(range(48))
        np.random.shuffle(deck)
        self.hands = {agent: [deck.pop() for _ in range(12)] for agent in self.agents}
        self.hands = {agent: sorted(hand) for agent, hand in self.hands.items()}
        self.tricks_02 = []
        self.tricks_13 = []
        self.last_trick_02 = []
        self.last_trick_13 = []
        self.table = []
        self.trump = np.random.randint(4)

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._clear_rewards()

    def step(self, action):
        try:
            self.failing_step(action)
        except (ValueError, IndexError) as e:
            # print(e)
            self.rewards[self.agent_selection] = -1
            self._accumulate_rewards()

    def failing_step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        card_to_play = action
        if card_to_play not in self.hands[agent]:
            raise IndexError(f"Agent {agent} tried to play {format_card(card_to_play)} but does not have it")

        if len(self.table) > 0:
            playing_suit = self.table[0] // 12
            legal_cards = [card for card in self.hands[agent] if card // 12 == playing_suit]
            if len(legal_cards) == 0:
                legal_cards = self.hands[agent]
            else:
                if card_to_play // 12 != playing_suit:
                    raise ValueError(f"Agent {agent} tried to play {format_card(card_to_play)} but must follow suit")

            winner = get_winner(self.table, self.trump)
            need_to_win = winner != len(self.table) - 2
            # print(f"Winner is {winner}, need to win: {need_to_win}")
            if need_to_win and not beats(card_to_play, self.table[winner], self.trump, playing_suit):
                can_win = any(beats(card, self.table[winner], self.trump, playing_suit) for card in legal_cards)
                if can_win:
                    raise ValueError(f"Agent {agent} tried to play {format_card(card_to_play)} but must win")

        # print(f"{agent} played {format_card(card_to_play)}")
        self.hands[agent].remove(card_to_play)

        self.table.append(card_to_play)

        if len(self.hands[agent]) == 0:
            self.terminations[agent] = True

        self._clear_rewards()
        if len(self.table) == 4:
            winner = get_winner(self.table, self.trump)
            winner_agent = self._agent_selector.agent_order[winner]
            # print(f"Winner is {winner_agent}")
            reward = sum(POINTS[card % 12] for card in self.table) + 1
            # print(f"reward is {reward}")
            if winner_agent in ("player_0", "player_2"):
                self.last_trick_02 = self.table
                self.tricks_02.append(self.table)
                self.rewards["player_0"] = reward / 2
                self.rewards["player_2"] = reward / 2
                self.rewards["player_1"] = -reward / 2
                self.rewards["player_3"] = -reward / 2
            else:
                self.last_trick_13 = self.table
                self.tricks_13.append(self.table)
                self.rewards["player_0"] = -reward / 2
                self.rewards["player_2"] = -reward / 2
                self.rewards["player_1"] = reward / 2
                self.rewards["player_3"] = reward / 2
            self.table = []

            new_order = self._agent_selector.agent_order[winner:] + self._agent_selector.agent_order[:winner]
            self._agent_selector.reinit(new_order)

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()


if __name__ == "__main__":
    import pettingzoo.test
    pettingzoo.test.api_test(ButifarraEnv(), num_cycles=10)
