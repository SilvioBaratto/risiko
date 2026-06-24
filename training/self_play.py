"""Self-play training loop with checkpoint rotation and resume."""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tqdm import tqdm

from src.agents.base import Agent
from src.agents.llm_opponent import LLMOpponent
from src.agents.player_config import load_profiles_from_yaml
from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.config import TrainingConfig
from src.env import RisikoEnv
from src.models.actor_critic import ActorCritic
from src.models.graph_actor_critic import GraphSAGEActorCritic
from src.models.ppo import PPOTrainer
from src.models.replay_buffer import RolloutBuffer
from src.models.utils import get_obs_dim
from src.multi_agent import GameResult, MultiAgentRunner
from src.tb_logger import TensorBoardLogger
from src.utils.constants import ACTION_DIMS, NUM_TERRITORIES
from src.utils.log import get_logger
from src.utils.seed import set_global_seeds
from training.early_stopping import EarlyStopper

_log = get_logger("train")

_LEARNER_ID: int = 0
_OPPONENT_ID: int = 1


def _build_llm_opponents(
    n_players: int,
    llm_profiles_path: str | None,
    llm_model: str | None = None,
) -> list[LLMOpponent | None]:
    """Build N-1 opponent slots from a profile YAML or return None fillers.

    Args:
        n_players: Total number of players (learner + opponents).
        llm_profiles_path: Path to a YAML profile list, or ``None`` for
            self-play / random-filler mode.
        llm_model: If set, override every profile's model with this name,
            keeping each slot's temperature / top_p / strategy_hint.

    Returns:
        A list of ``n_players - 1`` items: ``LLMOpponent`` instances when
        *llm_profiles_path* is set, ``None`` values otherwise.
    """
    n_opponents = n_players - 1
    if not llm_profiles_path:
        return [None] * n_opponents
    profiles = load_profiles_from_yaml(Path(llm_profiles_path))
    if llm_model:
        profiles = [dataclasses.replace(p, model=llm_model) for p in profiles]
    opponents: list[LLMOpponent | None] = [
        LLMOpponent(player_config=profiles[i % len(profiles)]) for i in range(n_opponents)
    ]
    _log.info(
        "LLM mode: loaded %d opponents from %s (model=%s)",
        n_opponents,
        llm_profiles_path,
        llm_model or "per-profile",
    )
    return opponents


class SelfPlayTrainer:
    """Train a PPO agent via self-play with periodic opponent promotion."""

    def __init__(
        self,
        cfg: TrainingConfig,
        checkpoint_dir: Path = Path("models"),
        log_dir: Path | None = None,
        max_turns: int | None = None,
    ) -> None:
        """Initialise trainer, networks, opponent, and logging.

        Args:
            cfg: Training hyperparameters.
            checkpoint_dir: Directory for checkpoints.
            log_dir: TensorBoard log directory.
            max_turns: Override for ``cfg.max_turns``. Falls back to
                ``cfg.max_turns`` when ``None``.
        """
        self._cfg = cfg
        self._device = self._resolve_device(cfg.device)
        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir or self._default_log_dir()
        self._max_turns = max_turns if max_turns is not None else cfg.max_turns
        self._rng = set_global_seeds(cfg.seed)
        self._env = RisikoEnv(n_players=cfg.self_play.n_players, reward_config=cfg.reward)
        self._trainer, self._agent = self._build_trainer()
        self._opponent_net = self._build_opponent_net()
        self._opponent_agent = self._build_opponent_agent()
        self._llm_opponents: list[LLMOpponent] | None = self._init_llm_opponents()
        self._logger = TensorBoardLogger(self._log_dir)
        self._episode = 0
        self._best_metric_value = 0.0
        self._early_stopper = self._build_early_stopper()
        self._buffer: RolloutBuffer | None = None
        sp = cfg.self_play
        self._curriculum_mode: str | None = sp.curriculum_mode if sp.curriculum_enabled else None
        self._curriculum_results: deque[float] = deque(maxlen=sp.curriculum_window)
        self._last_balanced: bool = False  # was the last episode a balanced (non-curriculum) one
        # Difficulty knob: territory mode → opponent_territories (grows to the
        # even share = full game); army mode → learner army multiplier (shrinks
        # to 1.0 = balanced full game).
        if self._curriculum_mode == "army":
            self._curriculum_val: float = sp.curriculum_army_start
            _log.info("curriculum ON (army): start multiplier=%.1f → 1.0", self._curriculum_val)
        elif self._curriculum_mode == "territory":
            self._curriculum_val = float(sp.curriculum_start)
            _log.info(
                "curriculum ON (territory): start opponent_territories=%d → full=%d",
                int(self._curriculum_val),
                NUM_TERRITORIES // sp.n_players,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        checkpoint_dir: Path = Path("models"),
        log_dir: Path | None = None,
    ) -> SelfPlayTrainer:
        """Construct a SelfPlayTrainer from a checkpoint file (for warm-starting from BC).

        Args:
            checkpoint_path: Path to the checkpoint file (typically from BC pretraining).
            checkpoint_dir: Directory for saving new checkpoints.
            log_dir: TensorBoard log directory.

        Returns:
            A SelfPlayTrainer instance with the checkpoint's weights loaded.
        """
        payload = torch.load(checkpoint_path, weights_only=False)
        cfg = payload["config"]
        trainer = cls(cfg, checkpoint_dir, log_dir)
        trainer.load_checkpoint(Path(checkpoint_path))
        return trainer

    def update_step(self) -> dict[str, float]:
        """Execute a single PPO update step for testing purposes.

        Creates a minimal rollout buffer, collects one episode, and updates the network.
        Used primarily for integration testing (warm-start BC → one PPO step).
        """
        self._buffer = self._make_buffer()
        result = self._run_episode(self._buffer)
        if len(self._buffer) > 0:
            self._update_and_log(self._buffer, result)
            self._buffer.clear()
        return {}

    def train(self) -> None:
        """Run the self-play training loop."""
        self._buffer = self._make_buffer()
        self._maybe_resume()
        self._log_seed()
        if self._llm_opponents:
            mode = f"llm:{self._cfg.self_play.llm_profiles_path}"
        else:
            mode = "self_play"
        _log.info(
            "training start — episodes=%d n_players=%d lr=%g seed=%d device=%s mode=%s",
            self._cfg.total_timesteps,
            self._cfg.self_play.n_players,
            self._cfg.ppo.lr,
            self._cfg.seed,
            self._device,
            mode,
        )
        buffer = self._buffer
        last_result: GameResult | None = None
        pbar = tqdm(
            total=self._cfg.total_timesteps,
            initial=self._episode,
            unit="ep",
            desc="Training",
        )
        while self._episode < self._cfg.total_timesteps:
            self._set_rollout_seed()
            result = self._run_episode(buffer)
            last_result = result
            self._maybe_advance_curriculum(result)
            self._logger.log_game_result(result, player_id=_LEARNER_ID, episode=self._episode)
            updated = False
            if len(buffer) >= buffer.capacity:
                self._update_and_log(buffer, result)
                buffer.clear()
                updated = True
            self._maybe_evaluate_and_promote()
            self._maybe_checkpoint()
            stop = self._maybe_early_stop()
            self._episode += 1
            pbar.update(1)
            if updated:
                winner = result.winner if result else "?"
                pbar.set_postfix(buf=f"{len(buffer)}/{buffer.capacity}", winner=winner)
            if result.winner is not None:
                _log.info(
                    "episode=%d winner=player_%d turns=%d eliminations=%s",
                    self._episode,
                    result.winner,
                    result.n_turns,
                    result.elimination_order,
                )
            if stop:
                break
        pbar.close()
        if len(buffer) > 0 and last_result is not None:
            self._update_and_log(buffer, last_result)
        self._maybe_restore_best()
        self._logger.close()
        _log.info("training complete — total episodes=%d", self._episode)

    def save_checkpoint(self, path: Path | None = None) -> None:
        """Save a checkpoint including model, opponent, and rng state."""
        path = path or self._checkpoint_dir / f"checkpoint_{self._episode}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer_state": self._trainer.save_checkpoint(),
            "rng_state": self._capture_rng(),
            "episode": self._episode,
            "config": self._cfg,
            "opponent_state": self._opponent_net.state_dict(),
            "best_metric_value": self._best_metric_value,
        }
        if self._buffer is not None:
            payload["buffer_state"] = self._buffer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> None:
        """Resume training from a checkpoint file."""
        payload = torch.load(path, weights_only=False)
        self._trainer.load_checkpoint(payload["trainer_state"])
        if "opponent_state" in payload:
            self._opponent_net.load_state_dict(payload["opponent_state"])
        self._episode = payload.get("episode", 0)
        self._best_metric_value = payload.get("best_metric_value", 0.0)
        self._restore_rng(payload.get("rng_state", {}))
        buffer_state = payload.get("buffer_state")
        if buffer_state is not None and self._buffer is not None:
            self._buffer.load_state_dict(buffer_state, device=self._device)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _resolve_device(self, device_str: str) -> str:
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

    def _default_log_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("results") / "runs" / f"self_play_{stamp}"

    def _build_trainer(self) -> tuple[PPOTrainer, PPOAgent]:
        net = self._build_net()
        reference_net = self._build_reference_net()
        trainer = PPOTrainer(net, self._cfg.ppo, reference_net=reference_net)
        agent = PPOAgent(net, device=self._device)
        return trainer, agent

    def _build_reference_net(self) -> ActorCritic | None:
        """Load a frozen BC policy to anchor PPO toward (RLHF-style).

        Enabled when ``ppo.kl_ref_coef > 0`` and ``bc.output_path`` exists.
        """
        if self._cfg.ppo.kl_ref_coef <= 0:
            return None
        path = Path(self._cfg.bc.output_path)
        if not path.exists():
            _log.warning("kl_ref_coef>0 but BC reference %s missing — no anchor", path)
            return None
        payload = torch.load(path, weights_only=False)
        ref = self._build_net()
        ref.load_state_dict(payload["trainer_state"]["model"])
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        _log.info(
            "KL-ref anchor: loaded BC reference %s (coef=%g)",
            path,
            self._cfg.ppo.kl_ref_coef,
        )
        return ref

    def _build_net(self) -> ActorCritic:
        # GraphSAGEActorCritic is interface-compatible (same forward /
        # get_action_and_value / action_dims), so it stands in for ActorCritic.
        arch = getattr(self._cfg.network, "arch", "mlp")
        net_cls = GraphSAGEActorCritic if arch == "graphsage" else ActorCritic
        net = net_cls(
            obs_dim=get_obs_dim(),
            hidden_size=self._cfg.network.hidden_sizes[0],
            num_layers=len(self._cfg.network.hidden_sizes),
            action_dims=ACTION_DIMS,
        ).to(self._device)
        return cast(ActorCritic, net)

    def _build_opponent_net(self) -> ActorCritic:
        net = self._build_net()
        net.load_state_dict(self._agent._net.state_dict())
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def _build_opponent_agent(self) -> PPOAgent:
        return PPOAgent(self._opponent_net, device=self._device)

    def _make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(capacity=self._cfg.ppo.n_steps, device=self._device)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _set_rollout_seed(self) -> None:
        set_global_seeds(self._cfg.seed + self._episode)

    def _init_llm_opponents(self) -> list[LLMOpponent] | None:
        """Load LLM opponents from profiles file, or return None for self-play mode."""
        slots = _build_llm_opponents(
            self._cfg.self_play.n_players,
            self._cfg.self_play.llm_profiles_path,
            self._cfg.self_play.llm_model,
        )
        if not self._cfg.self_play.llm_profiles_path:
            return None
        return slots  # type: ignore[return-value]

    def _build_agents(self) -> list[Agent]:
        """Return the ordered agent list for one episode.

        LLM mode: slot 0 = learner, slots 1..N-1 = LLM opponents.
        Self-play mode: slot 0 = learner, slot 1 = frozen PPO copy, slots 2..N-1 = random.
        """
        if self._llm_opponents is not None:
            return [self._agent, *self._llm_opponents]
        agents: list[Agent] = [self._agent, self._opponent_agent]
        n_filler = self._cfg.self_play.n_players - 2
        agents += [RandomAgent() for _ in range(n_filler)]
        return agents

    def _run_episode(self, buffer: RolloutBuffer) -> GameResult:
        """Play one episode and append learner transitions to *buffer*."""
        # Mix in balanced full-game episodes so the policy trains on the eval
        # distribution (not only near-won curriculum states). These episodes are
        # flagged so they don't count toward curriculum-stage promotion.
        frac = self._cfg.self_play.curriculum_balanced_fraction
        self._last_balanced = (
            self._curriculum_mode is not None and frac > 0 and self._rng.random() < frac
        )
        self._env.curriculum = None if self._last_balanced else self._curriculum_spec()
        agents = self._build_agents()
        runner = MultiAgentRunner(
            self._env,
            agents,
            max_turns=self._max_turns,
            stop_on_eliminated=_LEARNER_ID,
        )
        result = runner.run_game(seed=self._cfg.seed + self._episode)
        self._buffer_learner_transitions(result, buffer)
        return result

    def _curriculum_spec(self) -> dict[str, float] | None:
        """Env curriculum dict for the current stage, or None when off."""
        if self._curriculum_mode == "territory":
            return {
                "advantaged_player": _LEARNER_ID,
                "opponent_territories": int(self._curriculum_val),
            }
        if self._curriculum_mode == "army":
            return {"advantaged_player": _LEARNER_ID, "army_advantage": float(self._curriculum_val)}
        return None

    def _maybe_advance_curriculum(self, result: GameResult) -> None:
        """Promote to a harder curriculum stage once the learner reliably wins."""
        if self._curriculum_mode is None or self._last_balanced:
            return  # balanced full-game episodes don't gate stage promotion
        self._curriculum_results.append(1.0 if result.winner == _LEARNER_ID else 0.0)
        sp = self._cfg.self_play
        if len(self._curriculum_results) < sp.curriculum_window:
            return
        win_rate = sum(self._curriculum_results) / len(self._curriculum_results)
        if win_rate < sp.curriculum_promote_threshold:
            return
        self._curriculum_results.clear()
        if self._curriculum_mode == "territory":
            self._curriculum_val += sp.curriculum_step
            if self._curriculum_val >= NUM_TERRITORIES // sp.n_players:
                self._curriculum_mode = None
                _log.info("curriculum: win_rate=%.2f → full-game difficulty (off)", win_rate)
            else:
                _log.info(
                    "curriculum: win_rate=%.2f → opponent_territories=%d",
                    win_rate,
                    int(self._curriculum_val),
                )
        else:  # army
            self._curriculum_val -= sp.curriculum_army_step
            if self._curriculum_val <= 1.0:
                self._curriculum_mode = None
                _log.info("curriculum: win_rate=%.2f → balanced full game (off)", win_rate)
            else:
                _log.info(
                    "curriculum: win_rate=%.2f → army_multiplier=%.1f",
                    win_rate,
                    self._curriculum_val,
                )

    def _buffer_learner_transitions(self, result: GameResult, buffer: RolloutBuffer) -> None:
        """Copy learner transitions straight from trajectory into the rollout buffer.

        log_prob and value were captured at action time via ``act_with_meta``
        (see ``MultiAgentRunner._pick_action``), so no recomputation is needed —
        and recomputation would corrupt PPO's importance ratio if the masks
        weren't replayed identically. The full ``legal_actions`` list flows
        through so the update step can rebuild a per-head mask consistent with
        the one used at sampling time.

        Forced-skip transitions (empty legal_actions → NaN log_prob/value) are
        skipped: they aren't policy choices, so they shouldn't contribute to
        the gradient.

        The learner's *last* transition is marked ``done=True`` so GAE stops the
        value bootstrap at the game boundary — the buffer concatenates many
        games, so without a done flag advantages bleed across them. That
        terminal transition also receives the game's outcome reward: a win is
        already encoded by the env's ``sparse_win`` on the winning move; a loss
        and a draw are not (the loss fires on the *opponent's* step, and a
        capped draw fires nothing), so we add ``sparse_loss`` or the graded
        territory ``margin`` reward respectively. Without this the value head
        never sees a terminal win/loss target and its loss collapses to ~0.
        """
        import math as _math

        learner_traj = [
            t
            for t in result.trajectories
            if int(t.obs["current_player"]) == _LEARNER_ID
            and not _math.isnan(t.log_prob)
            and not _math.isnan(t.value)
        ]
        total_learner = sum(
            1 for t in result.trajectories if int(t.obs["current_player"]) == _LEARNER_ID
        )
        terminal_idx = len(learner_traj) - 1
        terminal_extra = self._terminal_outcome_reward(result)

        added = 0
        for i, t in enumerate(learner_traj):
            if len(buffer) >= buffer.capacity:
                break
            is_terminal = i == terminal_idx
            reward = t.reward + (terminal_extra if is_terminal else 0.0)
            added += 1
            buffer.add(
                t.obs,
                {k: torch.tensor(v, device=self._device) for k, v in t.action.items()},
                reward,
                is_terminal,
                t.value,
                t.log_prob,
                t.legal_actions,
            )
        _log.debug(
            "episode=%d transitions: learner=%d added=%d buffer=%d/%d terminal_extra=%.3f",
            self._episode,
            total_learner,
            added,
            len(buffer),
            buffer.capacity,
            terminal_extra,
        )

    def _terminal_outcome_reward(self, result: GameResult) -> float:
        """Outcome reward added to the learner's final transition.

        - learner won  → 0.0 (``sparse_win`` already on the winning move)
        - learner eliminated → ``sparse_loss`` (game may stop here before a
          single winner emerges, so check elimination before the draw branch)
        - draw (cap hit, learner still alive, no winner) → graded territory margin
        """
        cfg = self._env.reward_config
        if result.winner == _LEARNER_ID:
            return 0.0
        if _LEARNER_ID in result.elimination_order:
            return cfg.sparse_loss
        if result.winner is None:
            return cfg.terminal_margin_weight * self._env.territory_margin(_LEARNER_ID)
        return cfg.sparse_loss

    # ------------------------------------------------------------------
    # PPO update and logging
    # ------------------------------------------------------------------

    def _update_and_log(self, buffer: RolloutBuffer, result: GameResult | None) -> None:
        del result  # game results are logged per-episode in the main loop
        if len(buffer) == 0:
            return
        self._compute_advantages(buffer)
        metrics = self._trainer.update(buffer)
        _log.info(
            "ppo update ep=%d | kl=%.4f clip=%.3f policy_loss=%.4f value_loss=%.4f entropy=%.4f",
            self._episode,
            metrics.get("approx_kl", 0),
            metrics.get("clip_fraction", 0),
            metrics.get("policy_loss", 0),
            metrics.get("value_loss", 0),
            metrics.get("entropy_loss", 0),
        )
        self._logger.log_training_step(metrics, episode=self._episode)
        buffer.clear()

    def _compute_advantages(self, buffer: RolloutBuffer) -> None:
        with torch.no_grad():
            last_obs = buffer.observations[-1]
            obs_t = self._agent._prepare_obs(last_obs)
            _logits, last_value = self._agent._net(obs_t)
        buffer.compute_advantages(
            next_value=torch.tensor([last_value.item()]),
            next_done=torch.tensor([0.0]),
            gamma=self._cfg.ppo.gamma,
            gae_lambda=self._cfg.ppo.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Evaluation and opponent promotion
    # ------------------------------------------------------------------

    def _maybe_evaluate_and_promote(self) -> None:
        if self._llm_opponents is not None:
            return  # LLM opponents are fixed; no promotion needed
        if self._episode == 0:
            return  # nothing trained yet — skip the pre-training evaluation
        if self._episode % self._cfg.self_play.opponent_update_freq != 0:
            return
        win_rate = self._evaluate_against_opponent()
        threshold = self._cfg.self_play.promote_threshold
        promoted = win_rate > threshold
        _log.info(
            "eval ep=%d | win_rate=%.3f threshold=%.2f → %s",
            self._episode,
            win_rate,
            threshold,
            "PROMOTE" if promoted else "keep",
        )
        if promoted:
            self._promote_current_to_opponent()

    def _evaluate_against_opponent(self) -> float:
        from training.evaluate import evaluate_agents

        result = evaluate_agents(
            self._agent,
            self._opponent_agent,
            n_games=self._cfg.self_play.eval_games,
            n_players=2,
            seed=self._cfg.seed + self._episode,
            max_turns=self._max_turns,
        )
        return result.win_rate_a

    def _promote_current_to_opponent(self) -> None:
        self._opponent_net.load_state_dict(self._agent._net.state_dict())
        _log.info("opponent promoted at episode=%d", self._episode)

    # ------------------------------------------------------------------
    # Early stopping (plateau detection on win-rate vs a fixed baseline)
    # ------------------------------------------------------------------

    def _build_early_stopper(self) -> EarlyStopper | None:
        es = self._cfg.early_stop
        if not es.enabled:
            return None
        return EarlyStopper(patience=es.patience, min_delta=es.min_delta, window=es.window)

    def _maybe_early_stop(self) -> bool:
        """Evaluate vs a fixed baseline on schedule; return True to stop.

        Win-rate against the training opponent is circular and (in 6p) ~0 for a
        long time, so the plateau signal is win-rate against a fixed
        ``RandomAgent`` reference — a clean, monotone-ish generalisation curve
        that costs no LLM calls. The best checkpoint is snapshotted so training
        can restore the peak (self-play agents tend to degrade past it).
        """
        es = self._cfg.early_stop
        if self._early_stopper is None:
            return False
        if self._episode == 0 or self._episode % es.eval_every != 0:
            return False
        win_rate = self._evaluate_against_random()
        stop = self._early_stopper.update(win_rate)
        self._logger.log_scalar("early_stop/win_rate_vs_random", win_rate, self._episode)
        self._logger.log_scalar(
            "early_stop/smoothed_best", self._early_stopper.best_value, self._episode
        )
        _log.info(
            "early-stop eval ep=%d | win_rate_vs_random=%.3f smoothed_best=%.3f bad=%d/%d%s",
            self._episode,
            win_rate,
            self._early_stopper.best_value,
            self._early_stopper.num_bad_evals,
            es.patience,
            " — NEW BEST" if self._early_stopper.is_best else "",
        )
        if self._early_stopper.is_best:
            self.save_checkpoint(self._checkpoint_dir / "best.pt")
        if stop:
            _log.info(
                "early stopping at ep=%d — no win-rate gain in %d evals (best=%.3f)",
                self._episode,
                es.patience,
                self._early_stopper.best_value,
            )
        return stop

    def _evaluate_against_random(self) -> float:
        """Win-rate of the learner (player 0) vs an all-``RandomAgent`` board."""
        from src.agents.random_agent import RandomAgent
        from training.evaluate import evaluate_agents

        result = evaluate_agents(
            self._agent,
            RandomAgent(seed=self._cfg.seed),
            n_games=self._cfg.early_stop.eval_games,
            n_players=self._cfg.self_play.n_players,
            seed=self._cfg.seed + self._episode,
            max_turns=self._max_turns,
        )
        return result.win_rate_a

    def _maybe_restore_best(self) -> None:
        """Reload the best checkpoint so the final model is the peak, not the last."""
        es = self._cfg.early_stop
        if self._early_stopper is None or not es.restore_best:
            return
        best_path = self._checkpoint_dir / "best.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
            _log.info(
                "restored best checkpoint from %s (best=%.3f)",
                best_path,
                self._early_stopper.best_value,
            )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self) -> None:
        if self._cfg.save_freq <= 0 or self._episode % self._cfg.save_freq != 0:
            return
        self.save_checkpoint()
        self.save_checkpoint(self._checkpoint_dir / "latest.pt")

    def _capture_rng(self) -> dict[str, Any]:
        return {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "env": self._env.state.rng.bit_generator.state,
        }

    def _restore_rng(self, rng_state: dict[str, Any]) -> None:
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "random" in rng_state:
            random.setstate(rng_state["random"])
        if "env" in rng_state:
            bg = np.random.PCG64()
            bg.state = rng_state["env"]
            self._env.state.rng = np.random.Generator(bg)

    def _maybe_resume(self) -> None:
        latest = self._checkpoint_dir / "latest.pt"
        if latest.exists():
            self.load_checkpoint(latest)

    def _log_seed(self) -> None:
        print(f"Training seed: {self._cfg.seed}")


def train_self_play(
    cfg: TrainingConfig,
    checkpoint_dir: Path = Path("models"),
    log_dir: Path | None = None,
) -> None:
    """Run a self-play training loop with logging and periodic checkpoints."""
    trainer = SelfPlayTrainer(cfg, checkpoint_dir, log_dir)
    trainer.train()
