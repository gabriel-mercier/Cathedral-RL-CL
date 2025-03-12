import sys
import numpy as np
import pygame

from .utils import GIFRecorder


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, recorder: GIFRecorder = None):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.recorder = recorder

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        piece_cycle = 0
        selected_piece = -1
        rotation = 0
        # Default position in center of board using the board size from the environment.
        board_size = self.env.unwrapped.board_size  # or self.env.unwrapped.board.board_size if defined there
        pos = (board_size // 2, board_size // 2)
        mousex, mousey = 0, 0  # Default mouse coordinates at (0, 0)
        action = -1

        while True:
            event = pygame.event.wait()
            recorder = self.recorder
            env = self.env

            if event.type == pygame.QUIT:
                if recorder is not None:
                    recorder.end_recording(env.unwrapped.screen)
                pygame.quit()
                pygame.display.quit()
                sys.exit()

            """ GET MOUSE INPUT """
            if pygame.mouse.get_focused():
                mousex_prev, mousey_prev = mousex, mousey
                mousex, mousey = pygame.mouse.get_pos()
                # Compute block size based on current window size and board dimensions
                window_width, window_height = env.unwrapped.window.get_size()
                block_size = min(window_width, window_height) / board_size

                if mousex != mousex_prev or mousey != mousey_prev:
                    # Convert mouse position (in pixels) to board indices
                    pos = (int(mousex // block_size), int(mousey // block_size))
                    # Optional debug prints:
                    # print(f"mousex: {mousex}, mousey: {mousey}")
                    # print(f"Board indices: {pos[0]}, {pos[1]}")

            """ FIND PLACED PIECES """
            unplaced = env.unwrapped.board.unplaced_pieces[agent]

            # Default piece choice: last piece in the cycle (largest piece)
            if selected_piece == -1:
                if len(unplaced) > 0:
                    selected_piece = unplaced[len(unplaced) - 1]
                else:
                    selected_piece = -1

            """ READ KEYBOARD INPUT """
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Cycle through pieces (from largest to smallest)
                    if len(unplaced) > 0:
                        piece_cycle = (piece_cycle + 1) % len(unplaced)
                        selected_piece = unplaced[piece_cycle]
                elif event.key == pygame.K_e:
                    # Rotate piece clockwise
                    rotations = 0
                    while rotations < 4:
                        rotation = (rotation - 90) % 360
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos, rotation
                        )
                        if act != -1:
                            break
                        rotations += 1
                elif event.key == pygame.K_q:
                    # Rotate piece counter-clockwise
                    rotations = 0
                    while rotations < 4:
                        rotation = (rotation + 90) % 360
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos, rotation
                        )
                        if act != -1:
                            break
                        rotations += 1
                elif event.key == pygame.K_RIGHT:
                    pos_test = pos
                    # Use board_size as the upper bound (max index is board_size-1)
                    while pos_test[0] < board_size - 1:
                        pos_test = (pos_test[0] + 1, pos_test[1])
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1 and env.unwrapped.board.is_legal(agent, act):
                            pos = pos_test
                            break
                elif event.key == pygame.K_LEFT:
                    pos_test = pos
                    while pos_test[0] > 0:
                        pos_test = (pos_test[0] - 1, pos_test[1])
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1 and env.unwrapped.board.is_legal(agent, act):
                            pos = pos_test
                            break
                elif event.key == pygame.K_UP:
                    pos_test = pos
                    while pos_test[1] > 0:
                        pos_test = (pos_test[0], pos_test[1] - 1)
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1 and env.unwrapped.board.is_legal(agent, act):
                            pos = pos_test
                            break
                elif event.key == pygame.K_DOWN:
                    pos_test = pos
                    while pos_test[1] < board_size - 1:
                        pos_test = (pos_test[0], pos_test[1] + 1)
                        act = env.unwrapped.board.reverse_action_map(
                            agent, selected_piece, pos_test, rotation
                        )
                        if act != -1 and env.unwrapped.board.is_legal(agent, act):
                            pos = pos_test
                            break

            """ GET PREVIEW ACTION """
            action_prev = env.unwrapped.board.reverse_action_map(
                agent, selected_piece, pos, rotation
            )

            env.unwrapped.board.clear_previews()

            """ CLEAR ACTION PREVIEW FOR ILLEGAL MOVES """
            if action_prev != -1 and env.unwrapped.board.is_legal(agent, action_prev):
                env.unwrapped.board.preview_turn(agent, action_prev)

                """ UPDATE DISPLAY with previewed move """
                env.render()
                pygame.display.update()
                if recorder is not None:
                    recorder.capture_frame(env.unwrapped.screen)

                action = action_prev  # Store the latest legal action

            if action != -1:
                """PICK UP / PLACE A PIECE"""
                if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN):
                    env.unwrapped.board.clear_previews()
                    return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping
