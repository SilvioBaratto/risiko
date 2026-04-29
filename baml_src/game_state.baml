// Domain types for the Risiko (Risk) LLM opponent.
// Mirrors the env observation dict in a human-readable shape.

// ------------------------------------------------------------------
// Enums
// ------------------------------------------------------------------

enum Phase {
  TRADE        @description("Trade cards for reinforcements (optional)")
  REINFORCE    @description("Place reinforcements on owned territories")
  ATTACK       @description("Attack adjacent enemy territories")
  CAPTURE_MOVE @description("Move armies into newly captured territory")
  FORTIFY      @description("Move armies between owned territories (optional)")
}

// ------------------------------------------------------------------
// Supporting classes
// ------------------------------------------------------------------

class TerritorySnapshot {
  id int
  name string
  owner int
  armies int
  continent string
}

class CardInfo {
  index int
  symbol string
}

class LegalAction {
  action_type string
  description string
}

// ------------------------------------------------------------------
// Main snapshot class
// ------------------------------------------------------------------

class GameStateSnapshot {
  current_player int
  phase Phase
  territories TerritorySnapshot[]
  cards CardInfo[]
  reinforcements_remaining int
  trade_count int
  turn_capture bool
  n_players int
  eliminated int[]
  continent_control int[]
  legal_actions LegalAction[]
}

// ------------------------------------------------------------------
// Output action class
// ------------------------------------------------------------------

class RisikoAction {
  action_type string
  param_a int
  param_b int
  param_c int
  param_d int
  reasoning string? @description("Short tactical reasoning (optional)")
}

// ------------------------------------------------------------------
// LLM function
// ------------------------------------------------------------------

function GenerateRisikoAction(state: GameStateSnapshot) -> RisikoAction {
  client "Model"

  prompt #"
    You are a tactical advisor for the board game Risiko (Risk).
    The current player is Player {{ state.current_player }}.
    The game phase is: {{ state.phase }}.

    Board state (territories):
    {% for t in state.territories -%}
      - {{ t.name }} (id={{ t.id }}, continent={{ t.continent }}): owned by Player {{ t.owner }} with {{ t.armies }} armies
    {% endfor %}

    {% if state.continent_control | length > 0 -%}
    Continent control (by continent index 0..5):
    {{ state.continent_control }}
    {% endif %}

    {% if state.cards | length > 0 -%}
    Your cards:
    {% for c in state.cards -%}
      - Card {{ c.index }}: {{ c.symbol }}
    {% endfor %}
    {% else -%}
    You have no cards.
    {% endif %}

    Trade count so far: {{ state.trade_count }}
    Reinforcements remaining this phase: {{ state.reinforcements_remaining }}
    Players eliminated: {{ state.eliminated }}

    Legal actions for the current phase:
    {% for a in state.legal_actions -%}
      - {{ a.description }}
    {% endfor %}

    Choose exactly ONE legal action and return it using the structured output format below.
    Respect legality constraints: only pick from the listed actions, use valid territory IDs,
    and ensure you do not violate phase rules (e.g. only trade before attacking).

    {{ ctx.output_format }}
  "#
}
