negative_inf = float("-inf")
positive_inf = float("+inf")

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def get_outer_layer_cells(game):
    board_outer_layer = []
    for w in range(game.width):
        for h in range(game.height):
            if w == 0 or w == game.width - 1 or h == 0 or h == game.height - 1:
                board_outer_layer.append((w, h))
    return board_outer_layer                

def get_inner_layer_cells(game, center_w_high, center_w_low, center_h_high, center_h_low):
    board_inner_layer = []
    for w in range(game.width):
        for h in range(game.height):
            if (w < center_w_high and w > center_w_low) and (h < center_h_high and h > center_h_low):
                board_inner_layer.append((w, h))
    return board_inner_layer 

def get_middle_layer_cells(game, board_inner_layer, board_outer_layer):
    board_middle_layer = []
    for w in range(game.width):
        for h in range(game.height):
            if (w, h) not in board_outer_layer and (w, h) not in board_inner_layer:
                board_middle_layer.append((w, h))
    return board_middle_layer

def custom_score(game, player):
    """
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return negative_inf

    if game.is_winner(player):
        return positive_inf

    center_w_low = int(game.width / 2) - 1
    center_w_high = int(game.width / 2) + 1

    center_h_low = int(game.height / 2) - 1
    center_h_high = int(game.height / 2) + 1

    board_outer_layer = get_outer_layer_cells(game)
    board_inner_layer = get_inner_layer_cells(game, center_w_high, center_w_low, center_h_high, center_h_low)
    board_middle_layer = get_middle_layer_cells(game, board_inner_layer, board_outer_layer)

    current_player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    current_player_score = 0
    opponent_score = 0

    weight_inner = 3
    weight_middle = 2
    weight_outer = 1

    current_player_score += weight_inner * len([m for m in current_player_moves if m in board_inner_layer])
    current_player_score += weight_middle * len([m for m in current_player_moves if m in board_middle_layer])
    current_player_score += weight_outer * len([m for m in current_player_moves if m in board_outer_layer])

    opponent_score += weight_inner * len([m for m in opponent_moves if m in board_inner_layer])
    opponent_score += weight_middle * len([m for m in opponent_moves if m in board_middle_layer])
    opponent_score += weight_outer * len([m for m in opponent_moves if m in board_outer_layer])

    return (float(current_player_score - opponent_score) + float(len(current_player_moves) - len(opponent_moves)))

def custom_score_2(game, player):
    """
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return negative_inf

    if game.is_winner(player):
        return positive_inf

    agent_y, agent_x = game.get_player_location(player)
    opponent_y, opponent_x = game.get_player_location(game.get_opponent(player))
    
    width = game.width / 2
    height = game.height / 2

    euclidean_current_player = float((width - agent_y) ** 2 + (height - agent_x) ** 2)
    euclidean_opponent = float((width - opponent_y) ** 2 + (height - opponent_x) ** 2)

    return (euclidean_current_player - euclidean_opponent)


def custom_score_3(game, player):
    """
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return negative_inf

    if game.is_winner(player):
        return positive_inf

    current_player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return (float(current_player_moves) - float(opponent_moves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass

        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        final_move = (-1, -1)
        
        if depth == 0:
            return final_move
        
        final_score = negative_inf    

        moves = game.get_legal_moves()
        
        for move in moves:
            score = self.minimax_min_value(game.forecast_move(move), depth - 1)
            if score > final_score:
                final_score = score
                final_move = move
  
        return final_move              
        
    def minimax_min_value(self, game, depth):
        
        final_score = positive_inf

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        if len(moves) == 0 or depth == 0:
            return self.score(game, self)

        for move in moves:
            score = self.minimax_max_value(game.forecast_move(move), depth - 1)
            if score < final_score:
                final_score = score

        return final_score

    def minimax_max_value(self, game, depth):
        
        final_score = negative_inf

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        if len(moves) == 0 or depth == 0:
            return self.score(game, self)

        for move in moves:
            score = self.minimax_min_value(game.forecast_move(move), depth - 1)
            if score > final_score:
                final_score = score

        return final_score 

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        try:
            depth = 0
            while True:
                depth += 1
                best_move = self.alphabeta(game, depth)
        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=negative_inf, beta=positive_inf):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        final_move = (-1, -1)
        final_score = negative_inf

        if depth == 0:
            return final_move

        for move in moves:
            score = self.alphabeta_min_value(game.forecast_move(move), depth - 1, float(alpha), float(beta))
            if score > final_score:
                final_score = score
                final_move = move

            if final_score >= beta:
                return final_move

            alpha = max(alpha, final_score)

        return final_move

    def alphabeta_min_value(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        if len(moves) == 0:
            return self.score(game, self)

        if depth == 0:
            return self.score(game, self)

        final_score = positive_inf

        for move in moves:
            score = self.alphabeta_max_value(game.forecast_move(move), depth - 1, float(alpha), float(beta))
            if score < final_score:
                final_score = score

            if final_score <= alpha:
                return final_score
            beta = min(beta, final_score)

        return final_score
    
    def alphabeta_max_value(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()

        if len(moves) == 0:
            return self.score(game, self)

        if depth == 0:
            return self.score(game, self)

        final_score = negative_inf

        for move in moves:
            score = self.alphabeta_min_value(game.forecast_move(move), depth - 1, float(alpha), float(beta))
            if score > final_score:
                final_score = score

            if final_score >= beta:
                return final_score
            alpha = max(alpha, final_score)

        return final_score    
