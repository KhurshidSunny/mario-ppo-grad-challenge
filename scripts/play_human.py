"""Optional keyboard play window (pygame). Training stays headless.

Uses a fixed-size camera window that follows the player (not a huge full-map window).

Controls:
  Left / A       move left
  Right / D      move right
  Up / W / Space  jump (hold Right + Jump to clear gaps)
  R              reset level
  Esc / Q        quit

Examples:
  python scripts/play_human.py
  python scripts/play_human.py --scale 3 --width 720 --height 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.game.constants import (
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_RIGHT_JUMP,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
)
from mario_rl.game.engine import GameEngine
from mario_rl.utils.render_frames import render_rgb


def read_action(pygame_mod) -> int:
    keys = pygame_mod.key.get_pressed()
    left = keys[pygame_mod.K_LEFT] or keys[pygame_mod.K_a]
    right = keys[pygame_mod.K_RIGHT] or keys[pygame_mod.K_d]
    jump = (
        keys[pygame_mod.K_UP]
        or keys[pygame_mod.K_w]
        or keys[pygame_mod.K_SPACE]
    )
    if right and jump:
        return ACTION_RIGHT_JUMP
    if left and jump:
        return ACTION_JUMP
    if left:
        return ACTION_LEFT
    if right:
        return ACTION_RIGHT
    if jump:
        return ACTION_JUMP
    return ACTION_NOOP


def status_text(engine: GameEngine) -> str:
    if engine.reached_goal_flag:
        return "GOAL — press R to replay"
    if not engine.alive:
        return f"DEAD ({engine.death_cause}) — press R"
    return "ALIVE"


def crop_camera(
    frame: np.ndarray,
    engine: GameEngine,
    *,
    scale: int,
    view_w: int,
    view_h: int,
) -> np.ndarray:
    """Center a fixed viewport on the player inside the full-level RGB frame."""
    full_h, full_w = frame.shape[:2]
    view_w = min(view_w, full_w)
    view_h = min(view_h, full_h)

    cx = int((engine.player.x + PLAYER_WIDTH / 2.0) * scale)
    cy = int((engine.player.y + PLAYER_HEIGHT / 2.0) * scale)

    left = int(cx - view_w / 2)
    top = int(cy - view_h / 2)
    left = max(0, min(left, full_w - view_w))
    top = max(0, min(top, full_h - view_h))

    return frame[top : top + view_h, left : left + view_w]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Mario Level 1 with keyboard")
    parser.add_argument(
        "--level",
        type=Path,
        default=ROOT / "levels" / "level_1.txt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Pixel scale for tiles (default 3 = compact window)",
    )
    parser.add_argument("--width", type=int, default=720, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=400, help="Window height in pixels")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "pygame is required for keyboard play.\n"
            "Install with: pip install pygame\n"
            "(Training/eval stay headless and do not need a window.)"
        ) from exc

    engine = GameEngine(args.level)
    engine.reset(seed=args.seed)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("Mario RL — human play (run + jump gaps | R reset | Esc quit)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    engine.reset(seed=args.seed)

        if engine.alive and not engine.reached_goal_flag:
            action = read_action(pygame)
            engine.step(action)

        full = render_rgb(engine, scale=args.scale)
        view = crop_camera(
            full,
            engine,
            scale=args.scale,
            view_w=args.width,
            view_h=args.height,
        )
        # Pad if level render is smaller than window (rare)
        if view.shape[0] != args.height or view.shape[1] != args.width:
            canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            canvas[:, :] = (24, 32, 48)
            h = min(args.height, view.shape[0])
            w = min(args.width, view.shape[1])
            canvas[:h, :w] = view[:h, :w]
            view = canvas

        surface = pygame.surfarray.make_surface(view.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

        hud1 = (
            f"{status_text(engine)}  x={engine.player.x:.0f}  steps={engine.steps}"
        )
        hud2 = (
            "Yellow=Mario  Brown=ground  Red=spikes (jump over)  "
            "Pink=enemy  Green=goal  |  Run+JUMP  R=reset  Esc=quit"
        )
        screen.blit(font.render(hud1, True, (240, 240, 240)), (8, 6))
        screen.blit(font.render(hud2, True, (220, 220, 220)), (8, 24))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
