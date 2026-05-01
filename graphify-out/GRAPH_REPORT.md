# Graph Report - .  (2026-05-01)

## Corpus Check
- Corpus is ~39,726 words - fits in a single context window. You may not need a graph.

## Summary
- 2084 nodes · 4719 edges · 72 communities detected
- Extraction: 52% EXTRACTED · 48% INFERRED · 0% AMBIGUOUS · INFERRED: 2279 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Agent Protocol & BAML Bridge|Agent Protocol & BAML Bridge]]
- [[_COMMUNITY_CLI & Checkpoint Infrastructure|CLI & Checkpoint Infrastructure]]
- [[_COMMUNITY_Actor-Critic & PPO Trainer|Actor-Critic & PPO Trainer]]
- [[_COMMUNITY_Multi-Agent Runner & TB Logger|Multi-Agent Runner & TB Logger]]
- [[_COMMUNITY_Seeding & Self-Play Tests|Seeding & Self-Play Tests]]
- [[_COMMUNITY_BAML Type Builder|BAML Type Builder]]
- [[_COMMUNITY_BAML Async Client|BAML Async Client]]
- [[_COMMUNITY_BAML Generated Types|BAML Generated Types]]
- [[_COMMUNITY_Rollout Buffer Tests|Rollout Buffer Tests]]
- [[_COMMUNITY_Actor-Critic Tests|Actor-Critic Tests]]
- [[_COMMUNITY_BAML Bridge Builders|BAML Bridge Builders]]
- [[_COMMUNITY_Observation Flatten & Stack|Observation Flatten & Stack]]
- [[_COMMUNITY_Board Constants Property Tests|Board Constants Property Tests]]
- [[_COMMUNITY_BAML Client Init & Agent Loader|BAML Client Init & Agent Loader]]
- [[_COMMUNITY_Cross-Module Integration|Cross-Module Integration]]
- [[_COMMUNITY_BAML Game-State Tests|BAML Game-State Tests]]
- [[_COMMUNITY_Env Core Tests|Env Core Tests]]
- [[_COMMUNITY_Training Pipeline Wiring|Training Pipeline Wiring]]
- [[_COMMUNITY_Visualization Tests|Visualization Tests]]
- [[_COMMUNITY_GAE Computation|GAE Computation]]
- [[_COMMUNITY_Action Masking & Distributions|Action Masking & Distributions]]
- [[_COMMUNITY_PPO Agent Wrapper|PPO Agent Wrapper]]
- [[_COMMUNITY_Actor-Critic Forward Pass|Actor-Critic Forward Pass]]
- [[_COMMUNITY_Game Recorder|Game Recorder]]
- [[_COMMUNITY_Game Rules Documentation|Game Rules Documentation]]
- [[_COMMUNITY_Requirements Tests|Requirements Tests]]
- [[_COMMUNITY_CLI Command Tests|CLI Command Tests]]
- [[_COMMUNITY_BAML Event Watchers|BAML Event Watchers]]
- [[_COMMUNITY_Self-Play Config|Self-Play Config]]
- [[_COMMUNITY_BAML Bridge Tests|BAML Bridge Tests]]
- [[_COMMUNITY_Training Scripts|Training Scripts]]
- [[_COMMUNITY_ConstantsEnv Test Bridge|Constants/Env Test Bridge]]
- [[_COMMUNITY_Baseline Tournament|Baseline Tournament]]
- [[_COMMUNITY_Agent Loader & Random Agent|Agent Loader & Random Agent]]
- [[_COMMUNITY_Strategy Analyzer|Strategy Analyzer]]
- [[_COMMUNITY_Replay Image Exporter|Replay Image Exporter]]
- [[_COMMUNITY_Snapshot Builder|Snapshot Builder]]
- [[_COMMUNITY_Action Legality Check|Action Legality Check]]
- [[_COMMUNITY_Evaluation Result|Evaluation Result]]
- [[_COMMUNITY_Wilson Confidence Interval|Wilson Confidence Interval]]
- [[_COMMUNITY_ASCII Board Renderer|ASCII Board Renderer]]
- [[_COMMUNITY_Matplotlib Board Renderer|Matplotlib Board Renderer]]
- [[_COMMUNITY_Observation Dimension|Observation Dimension]]
- [[_COMMUNITY_Network Config|Network Config]]
- [[_COMMUNITY_Config Loader|Config Loader]]
- [[_COMMUNITY_CLI Override Merger|CLI Override Merger]]
- [[_COMMUNITY_Device Resolver|Device Resolver]]
- [[_COMMUNITY_pettingzoo dependency|pettingzoo dependency]]
- [[_COMMUNITY_torch dependency|torch dependency]]
- [[_COMMUNITY_playwright dependency|playwright dependency]]
- [[_COMMUNITY_resolve_device fragment|resolve_device fragment]]
- [[_COMMUNITY_get_trade_value|get_trade_value]]
- [[_COMMUNITY_TestRequirements class|TestRequirements class]]
- [[_COMMUNITY_test_constants module|test_constants module]]
- [[_COMMUNITY_TestTerritoryNames|TestTerritoryNames]]
- [[_COMMUNITY_TestContinents|TestContinents]]
- [[_COMMUNITY_TestAdjacency|TestAdjacency]]
- [[_COMMUNITY_TestCards|TestCards]]
- [[_COMMUNITY_TestTradeValues|TestTradeValues]]
- [[_COMMUNITY_TestStartingArmies|TestStartingArmies]]
- [[_COMMUNITY_test_checkpoint module|test_checkpoint module]]
- [[_COMMUNITY_TestCheckpointManagerSave|TestCheckpointManagerSave]]
- [[_COMMUNITY_TestCheckpointManagerLoad|TestCheckpointManagerLoad]]
- [[_COMMUNITY_TestPPOTrainerCheckpointMethods|TestPPOTrainerCheckpointMethods]]
- [[_COMMUNITY_test_agents module|test_agents module]]
- [[_COMMUNITY_BadAgent fixture|BadAgent fixture]]
- [[_COMMUNITY_TestRandomAgent|TestRandomAgent]]
- [[_COMMUNITY_TestPPOAgent|TestPPOAgent]]
- [[_COMMUNITY_BaselineResult|BaselineResult]]
- [[_COMMUNITY_EvaluationResult fragment|EvaluationResult fragment]]
- [[_COMMUNITY_render_game module|render_game module]]
- [[_COMMUNITY_render_ascii fragment|render_ascii fragment]]

## God Nodes (most connected - your core abstractions)
1. `RisikoEnv` - 454 edges
2. `PPOConfig` - 169 edges
3. `RandomAgent` - 155 edges
4. `ActorCritic` - 146 edges
5. `RewardConfig` - 136 edges
6. `TrainingConfig` - 135 edges
7. `RolloutBuffer` - 121 edges
8. `Agent` - 112 edges
9. `GameResult` - 98 edges
10. `PPOTrainer` - 94 edges

## Surprising Connections (you probably didn't know these)
- `gymnasium dependency` --references--> `RisikoEnv`  [INFERRED]
  requirements.txt → src/env.py
- `typer dependency` --references--> `Typer CLI app`  [INFERRED]
  requirements.txt → risiko_rl/cli.py
- `Configuration & Reproducibility Requirements` --rationale_for--> `TrainingConfig`  [INFERRED]
  CLAUDE.md → src/config.py
- `BAML GameStateSnapshot` --semantically_similar_to--> `GameState`  [INFERRED] [semantically similar]
  baml_client/types.py → src/env.py
- `Global seeding utility properties.` --uses--> `RewardConfig`  [INFERRED]
  tests/test_utils.py → src/utils/reward_config.py

## Hyperedges (group relationships)
- **Risiko Turn Phase Sequence** — rules_reinforcements, rules_attack, rules_fortification [EXTRACTED 1.00]
- **Reinforcement Sources** — rules_42_territories, rules_continent_bonus, rules_card_trade [EXTRACTED 1.00]
- **BAML LLM opponent type contract** — baml_gamestatesnapshot, baml_risikoaction, baml_phase, baml_b [EXTRACTED 0.90]
- **Environment state/phase/observation triad** — env_risikoenv, env_gamestate, env_phase_constants [EXTRACTED 0.95]
- **CLI command surface (train/watch/evaluate/benchmark)** — cli_train, cli_watch, cli_evaluate, cli_benchmark, cli_app [EXTRACTED 0.95]
- **PPO training loop participants** — ppo_ppotrainer, actor_critic_actorcritic, replay_buffer_rolloutbuffer, gae_compute_gae [INFERRED 0.90]
- **Agent protocol implementations** — base_agent, random_agent_randomagent, llm_opponent_llmopponent, ppo_agent_ppoagent [EXTRACTED 1.00]
- **LLM action selection pipeline** — llm_opponent_llmopponent, baml_bridge_build_snapshot, baml_bridge_baml_action_to_dict, baml_bridge_is_legal [EXTRACTED 1.00]
- **PPO end-to-end pipeline (obs→flatten→actor_critic→buffer→GAE→PPO)** — src_flatten_obs, src_actor_critic, src_rollout_buffer, src_compute_gae, src_ppo_trainer, test_models_integration_test_end_to_end [EXTRACTED 1.00]
- **CLI commands tested through Typer runner (train/evaluate/watch/benchmark)** — src_cli_app, test_watch_test_watch_command, test_evaluate_cli_test_evaluate_command, test_benchmark_test_benchmark_command, test_config_test_cli_flags [EXTRACTED 0.95]
- **LLM opponent decision path: env obs → BAML snapshot → BAML action → env action dict** — src_llm_opponent, src_build_snapshot, src_baml_action_to_dict, src_is_legal, baml_client_module, test_llm_opponent_test_valid_action [EXTRACTED 0.95]
- **Training & evaluation pipeline** — self_play_module, monte_carlo_module, evaluate_module [INFERRED 0.90]
- **Checkpoint save/load/deterministic resume** — test_checkpoint_save, test_checkpoint_load, test_checkpoint_deterministic_resume [EXTRACTED 1.00]
- **Agent Protocol conformance suite** — test_agents_random_agent, test_agents_ppo_agent, test_agents_llm_protocol [EXTRACTED 1.00]

## Communities

### Community 0 - "Agent Protocol & BAML Bridge"
Cohesion: 0.01
Nodes (212): Agent, ActionType, Integer action type constants matching the environment., Agent, Agent protocol definition., Protocol for all Risiko agents., Select an action given observation and legal actions., Select an action given observation and legal actions.          Args: (+204 more)

### Community 1 - "CLI & Checkpoint Infrastructure"
Cohesion: 0.02
Nodes (153): train(), _config_from_dict(), _config_to_dict(), load(), Checkpoint save/resume infrastructure for training runs., Serialize a TrainingConfig to a plain dict., Deserialize a TrainingConfig from a plain dict., Save and load training checkpoints deterministically. (+145 more)

### Community 2 - "Actor-Critic & PPO Trainer"
Cohesion: 0.04
Nodes (105): ActorCritic, Shared-trunk actor-critic network for PPO., PPOTrainer, Proximal Policy Optimization (PPO) implementation from scratch., Compute explained variance., PPO clipped-surrogate loss trainer., Initialize trainer with network and hyperparameters.          Args:, Return trainer state for checkpointing. (+97 more)

### Community 3 - "Multi-Agent Runner & TB Logger"
Cohesion: 0.04
Nodes (66): GameResult, _ratio(), TensorBoard logging wrapper for Risiko RL training metrics., Return fraction of *territory_ids* owned by *player_id*., Write training metrics and episode statistics to TensorBoard., Create a logger backed by a SummaryWriter.          Args:             log_dir: D, Log scalar metrics from a PPO update step.          Args:             metrics: D, Log episode-level statistics.          Args:             result: Outcome of a si (+58 more)

### Community 4 - "Seeding & Self-Play Tests"
Cohesion: 0.04
Nodes (30): Tests for the self-play training loop., Log probs should be real (not NaN) and non-positive (prob <= 1)., PPO update changes network weights., Save and restore training state., Full training loop runs without crash., Opponent promotion copies current weights., Resume training from a checkpoint., SelfPlayTrainer initialises all components. (+22 more)

### Community 5 - "BAML Type Builder"
Cohesion: 0.03
Nodes (25): CardInfo(), CardInfoAst, CardInfoProperties, CardInfoViewer, GameStateSnapshot(), GameStateSnapshotAst, GameStateSnapshotProperties, GameStateSnapshotViewer (+17 more)

### Community 6 - "BAML Async Client"
Cohesion: 0.04
Nodes (23): BamlAsyncClient, BamlHttpRequestClient, BamlHttpStreamRequestClient, BamlStreamClient, get_log_level(), Get the log level for the BAML Python client., Set the log level for the BAML Python client, Set the log JSON mode for the BAML Python client. (+15 more)

### Community 7 - "BAML Generated Types"
Cohesion: 0.03
Nodes (57): CardInfo, GameStateSnapshot, LegalAction, RisikoAction, StreamState, TerritorySnapshot, all_succeeded(), CardInfo (+49 more)

### Community 8 - "Rollout Buffer Tests"
Cohesion: 0.06
Nodes (45): _make_buffer(), Tests for the Rollout Buffer., Adding beyond capacity raises IndexError., Create a RolloutBuffer with a small capacity for testing., Advantage computation contracts., compute_advantages sets advantages and returns., compute_advantages on empty buffer raises ValueError., Advantages shape matches buffer length. (+37 more)

### Community 9 - "Actor-Critic Tests"
Cohesion: 0.05
Nodes (42): _make_net(), _random_flat(), Tests for the Actor-Critic shared-trunk network., Raw logits are not probabilities (vary across dimensions)., Sampling and evaluation contract., get_action_and_value returns 4 tensors when sampling., Sampled action has the same keys as action_dims., Each sampled action value is a valid index for its space. (+34 more)

### Community 10 - "BAML Bridge Builders"
Cohesion: 0.05
Nodes (31): ActionTypeStr, baml_action_to_dict(), _build_cards(), _build_legal_actions(), build_snapshot(), _build_territories(), _describe_action(), is_legal() (+23 more)

### Community 11 - "Observation Flatten & Stack"
Cohesion: 0.05
Nodes (38): Stack a list of individual observations into batched tensors.      Args:, stack_obs(), _batch(), Tests for observation flattening and stacking utilities., Current player is one-hot encoded with 6 slots., Cards matrix is flattened unchanged., Continent control vector is flattened unchanged., Convert a numpy value (scalar or array) to a single-batch tensor. (+30 more)

### Community 12 - "Board Constants Property Tests"
Cohesion: 0.04
Nodes (36): Property-based tests for board constants and game data., No territory may be adjacent to itself., No duplicate adjacencies within a territory., Territories must have expected neighbor counts., Risk card deck properties., There must be exactly 44 cards., Card symbol counts must match standard deck., There must be exactly 2 wild cards. (+28 more)

### Community 13 - "BAML Client Init & Agent Loader"
Cohesion: 0.05
Nodes (46): agent_loader.load_agent, BAML sync client (b), BamlSyncClient, BAML CardInfo, BAML GameStateSnapshot, Inlined .baml source file_map, BAML LegalAction, BAML Phase enum (+38 more)

### Community 14 - "Cross-Module Integration"
Cohesion: 0.05
Nodes (43): baml_client, src.agents.baml_bridge.ActionType, src.agents.baml_bridge.baml_action_to_dict, src.multi_agent.GameResult, src.multi_agent.MultiAgentRunner, visualization.render_game.ReplayExporter, src.utils.reward_config.RewardConfig, src.env.RisikoEnv (+35 more)

### Community 15 - "BAML Game-State Tests"
Cohesion: 0.05
Nodes (28): env(), Tests for BAML-generated game-state types and GenerateRisikoAction., RisikoAction construction and optional reasoning., Can create a RisikoAction with all params and no reasoning., Can create a RisikoAction with optional reasoning., Phase enum values match environment phases., Convert RisikoEnv observation to GameStateSnapshot., Build a GameStateSnapshot from env observation and info. (+20 more)

### Community 16 - "Env Core Tests"
Cohesion: 0.07
Nodes (25): Core tests for the Risiko Gymnasium environment., Reinforcement computation and placement., Reinforcements remaining must be positive at turn start., Placing armies must reduce reinforcements remaining., Placing 0 armies must be an invalid action., Attack validation and dice resolution., Ending attack phase must advance to fortify., Attacking a non-adjacent territory must be invalid. (+17 more)

### Community 17 - "Training Pipeline Wiring"
Cohesion: 0.05
Nodes (36): src.models.actor_critic.ActorCritic, src.models.gae.compute_gae, training.evaluate.evaluate_agents, src.models.utils.flatten_obs, src.config.PPOConfig, src.models.ppo.PPOTrainer, src.models.replay_buffer.RolloutBuffer, training.self_play.SelfPlayTrainer (+28 more)

### Community 18 - "Visualization Tests"
Cohesion: 0.08
Nodes (23): TestRenderAscii, TestRenderMatplotlib, _copy_state(), _draw_continent_backgrounds(), _draw_edges(), _draw_legend(), _draw_nodes(), _get_layout() (+15 more)

### Community 19 - "GAE Computation"
Cohesion: 0.08
Nodes (19): compute_gae(), Generalized Advantage Estimation (GAE) implementation., Compute Generalized Advantage Estimation.      Args:         rewards: Rewards fo, Compute GAE advantages and returns from stored transitions., Tests for Generalized Advantage Estimation (GAE)., GAE with one step works., Zero rewards and values give zero advantages and returns., GAE works with 2D tensors (n_envs, n_steps). (+11 more)

### Community 20 - "Action Masking & Distributions"
Cohesion: 0.09
Nodes (30): _apply_mask, _build_distributions, ActorCritic, PolicyHeads, ActionType, ActionTypeStr, baml_action_to_dict, build_snapshot (+22 more)

### Community 21 - "PPO Agent Wrapper"
Cohesion: 0.14
Nodes (9): build_action_mask(), _fill_empty_heads(), _initialise_mask(), PPO agent wrapping ActorCritic with action masking., Convert a list of legal action dicts into per-head boolean masks.      Per-head, Select an action using the policy network with masking., Select an action and return log-prob / value tensors as well., _validate() (+1 more)

### Community 22 - "Actor-Critic Forward Pass"
Cohesion: 0.12
Nodes (12): _apply_mask(), _build_distributions(), PolicyHeads, Actor-Critic shared-trunk network for PPO training., Collection of independent categorical policy heads., Apply a boolean mask by setting masked logits to -inf.      Raises:         Valu, Create policy heads mapping trunk output to action logits., Compute logits for every action component. (+4 more)

### Community 23 - "Game Recorder"
Cohesion: 0.24
Nodes (2): _GameRecorder, TestGameRecorderEliminationDetection

### Community 24 - "Game Rules Documentation"
Cohesion: 0.16
Nodes (16): risiko, 42 Territories / 6 Continents, Phase 2: Attack, Risk Card Trade-In (escalating army reward), Game Components (board, tokens, cards, dice), Continent Bonus, Dice-Based Combat Resolution, Phase 3: Fortification (+8 more)

### Community 25 - "Requirements Tests"
Cohesion: 0.18
Nodes (6): Tests that runtime dependencies are listed in requirements.txt., Critical dependencies must be declared., baml-py must be in requirements.txt., baml-py must have a minimum version pin., baml_py must be importable at runtime., TestRequirements

### Community 26 - "CLI Command Tests"
Cohesion: 0.2
Nodes (11): src.agents.base.Agent, risiko_rl.cli.app, src.agents.llm_opponent.LLMOpponent, TestBenchmarkCommand, TestCliFlags, TestEvaluateCommand, TestConfiguration, TestFallback (+3 more)

### Community 27 - "BAML Event Watchers"
Cohesion: 0.25
Nodes (5): BlockEvent, EventCollectorInternal, InternalEventBindings, VarEvent, Protocol

### Community 28 - "Self-Play Config"
Cohesion: 0.5
Nodes (4): src.config.SelfPlayConfig, src.config.TrainingConfig, TestTrainingConfigDefaults, TestSelfPlayConfigFields

### Community 29 - "BAML Bridge Tests"
Cohesion: 0.5
Nodes (4): src.agents.baml_bridge, TestBuildCards, TestBuildLegalActions, TestBuildTerritories

### Community 30 - "Training Scripts"
Cohesion: 0.67
Nodes (4): training/evaluate.py, training/monte_carlo.py, training/self_play.py, training/strategy_analysis.py

### Community 31 - "Constants/Env Test Bridge"
Cohesion: 0.67
Nodes (1): src.utils.constants

### Community 32 - "Baseline Tournament"
Cohesion: 0.67
Nodes (3): training.monte_carlo.BaselineResult, training.monte_carlo.run_baseline_tournament, TestRunBaselineTournament

### Community 33 - "Agent Loader & Random Agent"
Cohesion: 0.67
Nodes (3): risiko_rl.agent_loader.load_agent, src.agents.random_agent.RandomAgent, TestLoadAgent

### Community 34 - "Strategy Analyzer"
Cohesion: 0.67
Nodes (3): StrategyAnalyzer, _bfs_distance, _phase_for_turn

### Community 35 - "Replay Image Exporter"
Cohesion: 1.0
Nodes (3): render_matplotlib, ReplayExporter, _state_to_image

### Community 36 - "Snapshot Builder"
Cohesion: 1.0
Nodes (2): src.agents.baml_bridge.build_snapshot, TestBuildSnapshot

### Community 37 - "Action Legality Check"
Cohesion: 1.0
Nodes (2): src.agents.baml_bridge.is_legal, TestIsLegal

### Community 39 - "Evaluation Result"
Cohesion: 1.0
Nodes (2): training.evaluate.EvaluationResult, TestEvaluationResult

### Community 40 - "Wilson Confidence Interval"
Cohesion: 1.0
Nodes (2): training.monte_carlo.wilson_ci, TestWilsonCI

### Community 41 - "ASCII Board Renderer"
Cohesion: 1.0
Nodes (2): visualization.render_game.render_ascii, TestRenderAscii

### Community 42 - "Matplotlib Board Renderer"
Cohesion: 1.0
Nodes (2): visualization.render_game.render_matplotlib, TestRenderMatplotlib

### Community 43 - "Observation Dimension"
Cohesion: 1.0
Nodes (2): src.models.utils.get_obs_dim, TestGetObsDim

### Community 44 - "Network Config"
Cohesion: 1.0
Nodes (2): src.config.NetworkConfig, TestNetworkConfigDefaults

### Community 45 - "Config Loader"
Cohesion: 1.0
Nodes (2): src.config.load_config, TestLoadConfig

### Community 46 - "CLI Override Merger"
Cohesion: 1.0
Nodes (2): src.config.merge_cli_overrides, TestMergeCliOverrides

### Community 47 - "Device Resolver"
Cohesion: 1.0
Nodes (2): src.config.resolve_device, TestResolveDevice

### Community 53 - "pettingzoo dependency"
Cohesion: 1.0
Nodes (1): pettingzoo dependency

### Community 54 - "torch dependency"
Cohesion: 1.0
Nodes (1): torch dependency

### Community 55 - "playwright dependency"
Cohesion: 1.0
Nodes (1): playwright dependency

### Community 56 - "resolve_device fragment"
Cohesion: 1.0
Nodes (1): resolve_device

### Community 57 - "get_trade_value"
Cohesion: 1.0
Nodes (1): get_trade_value

### Community 65 - "TestRequirements class"
Cohesion: 1.0
Nodes (1): TestRequirements

### Community 73 - "test_constants module"
Cohesion: 1.0
Nodes (1): tests/test_constants.py

### Community 74 - "TestTerritoryNames"
Cohesion: 1.0
Nodes (1): TestTerritoryNames

### Community 75 - "TestContinents"
Cohesion: 1.0
Nodes (1): TestContinents

### Community 76 - "TestAdjacency"
Cohesion: 1.0
Nodes (1): TestAdjacency

### Community 77 - "TestCards"
Cohesion: 1.0
Nodes (1): TestCards

### Community 78 - "TestTradeValues"
Cohesion: 1.0
Nodes (1): TestTradeValues

### Community 79 - "TestStartingArmies"
Cohesion: 1.0
Nodes (1): TestStartingArmies

### Community 80 - "test_checkpoint module"
Cohesion: 1.0
Nodes (1): tests/test_checkpoint.py

### Community 81 - "TestCheckpointManagerSave"
Cohesion: 1.0
Nodes (1): TestCheckpointManagerSave

### Community 82 - "TestCheckpointManagerLoad"
Cohesion: 1.0
Nodes (1): TestCheckpointManagerLoad

### Community 83 - "TestPPOTrainerCheckpointMethods"
Cohesion: 1.0
Nodes (1): TestPPOTrainerCheckpointMethods

### Community 84 - "test_agents module"
Cohesion: 1.0
Nodes (1): tests/test_agents.py

### Community 85 - "BadAgent fixture"
Cohesion: 1.0
Nodes (1): BadAgent

### Community 86 - "TestRandomAgent"
Cohesion: 1.0
Nodes (1): TestRandomAgent

### Community 87 - "TestPPOAgent"
Cohesion: 1.0
Nodes (1): TestPPOAgent

### Community 88 - "BaselineResult"
Cohesion: 1.0
Nodes (1): BaselineResult

### Community 89 - "EvaluationResult fragment"
Cohesion: 1.0
Nodes (1): EvaluationResult

### Community 90 - "render_game module"
Cohesion: 1.0
Nodes (1): visualization/render_game.py

### Community 91 - "render_ascii fragment"
Cohesion: 1.0
Nodes (1): render_ascii

## Ambiguous Edges - Review These
- `render_matplotlib` → `render_matplotlib`  [AMBIGUOUS]
  visualization/render_game.py · relation: references

## Knowledge Gaps
- **312 isolated node(s):** `Board renderer and replay exporter for Risiko games.`, `Return an ASCII representation of the board.      Args:         state: Dict with`, `Return a matplotlib Figure showing the board as a network graph.      Territorie`, `Record board states and export to an animated GIF.`, `Initialise an empty exporter.` (+307 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Game Recorder`** (16 nodes): `_GameRecorder`, `.build_result()`, `._detect_eliminations()`, `.detect_trade()`, `.__init__()`, `.record_action()`, `.record_step()`, `._resolve_winner()`, `._territory_counts()`, `.run_game()`, `TestGameRecorderEliminationDetection`, `._make_obs()`, `.test_detects_multiple_eliminations_same_step()`, `.test_detects_single_elimination()`, `.test_elimination_order_in_game_result()`, `.test_no_false_positives_on_repeat_obs()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Constants/Env Test Bridge`** (3 nodes): `src.utils.constants`, `test_env_core.py`, `test_env_extended.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Builder`** (2 nodes): `src.agents.baml_bridge.build_snapshot`, `TestBuildSnapshot`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Action Legality Check`** (2 nodes): `src.agents.baml_bridge.is_legal`, `TestIsLegal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Evaluation Result`** (2 nodes): `training.evaluate.EvaluationResult`, `TestEvaluationResult`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Wilson Confidence Interval`** (2 nodes): `training.monte_carlo.wilson_ci`, `TestWilsonCI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ASCII Board Renderer`** (2 nodes): `visualization.render_game.render_ascii`, `TestRenderAscii`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matplotlib Board Renderer`** (2 nodes): `visualization.render_game.render_matplotlib`, `TestRenderMatplotlib`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Observation Dimension`** (2 nodes): `src.models.utils.get_obs_dim`, `TestGetObsDim`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Network Config`** (2 nodes): `src.config.NetworkConfig`, `TestNetworkConfigDefaults`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Loader`** (2 nodes): `src.config.load_config`, `TestLoadConfig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CLI Override Merger`** (2 nodes): `src.config.merge_cli_overrides`, `TestMergeCliOverrides`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Device Resolver`** (2 nodes): `src.config.resolve_device`, `TestResolveDevice`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `pettingzoo dependency`** (1 nodes): `pettingzoo dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `torch dependency`** (1 nodes): `torch dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `playwright dependency`** (1 nodes): `playwright dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `resolve_device fragment`** (1 nodes): `resolve_device`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `get_trade_value`** (1 nodes): `get_trade_value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestRequirements class`** (1 nodes): `TestRequirements`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `test_constants module`** (1 nodes): `tests/test_constants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestTerritoryNames`** (1 nodes): `TestTerritoryNames`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestContinents`** (1 nodes): `TestContinents`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestAdjacency`** (1 nodes): `TestAdjacency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestCards`** (1 nodes): `TestCards`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestTradeValues`** (1 nodes): `TestTradeValues`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestStartingArmies`** (1 nodes): `TestStartingArmies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `test_checkpoint module`** (1 nodes): `tests/test_checkpoint.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestCheckpointManagerSave`** (1 nodes): `TestCheckpointManagerSave`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestCheckpointManagerLoad`** (1 nodes): `TestCheckpointManagerLoad`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestPPOTrainerCheckpointMethods`** (1 nodes): `TestPPOTrainerCheckpointMethods`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `test_agents module`** (1 nodes): `tests/test_agents.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BadAgent fixture`** (1 nodes): `BadAgent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestRandomAgent`** (1 nodes): `TestRandomAgent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `TestPPOAgent`** (1 nodes): `TestPPOAgent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BaselineResult`** (1 nodes): `BaselineResult`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `EvaluationResult fragment`** (1 nodes): `EvaluationResult`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `render_game module`** (1 nodes): `visualization/render_game.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `render_ascii fragment`** (1 nodes): `render_ascii`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `render_matplotlib` and `render_matplotlib`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `RisikoEnv` connect `Agent Protocol & BAML Bridge` to `CLI & Checkpoint Infrastructure`, `Actor-Critic & PPO Trainer`, `Multi-Agent Runner & TB Logger`, `Seeding & Self-Play Tests`, `BAML Bridge Builders`, `Board Constants Property Tests`, `BAML Game-State Tests`, `Env Core Tests`, `Visualization Tests`, `PPO Agent Wrapper`, `Game Recorder`?**
  _High betweenness centrality (0.336) - this node is a cross-community bridge._
- **Why does `TerritorySnapshotAst` connect `BAML Type Builder` to `CLI & Checkpoint Infrastructure`?**
  _High betweenness centrality (0.146) - this node is a cross-community bridge._
- **Why does `ActorCritic` connect `Actor-Critic & PPO Trainer` to `Agent Protocol & BAML Bridge`, `CLI & Checkpoint Infrastructure`, `Multi-Agent Runner & TB Logger`, `Seeding & Self-Play Tests`, `Actor-Critic Tests`, `PPO Agent Wrapper`, `Actor-Critic Forward Pass`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 420 inferred relationships involving `RisikoEnv` (e.g. with `BaselineResult` and `Monte Carlo baseline estimation with Wilson score confidence intervals.`) actually correct?**
  _`RisikoEnv` has 420 INFERRED edges - model-reasoned connections that need verification._
- **Are the 167 inferred relationships involving `PPOConfig` (e.g. with `TestSelfPlayTrainerInit` and `TestSelfPlayTrainerRunEpisode`) actually correct?**
  _`PPOConfig` has 167 INFERRED edges - model-reasoned connections that need verification._
- **Are the 149 inferred relationships involving `RandomAgent` (e.g. with `BaselineResult` and `Monte Carlo baseline estimation with Wilson score confidence intervals.`) actually correct?**
  _`RandomAgent` has 149 INFERRED edges - model-reasoned connections that need verification._