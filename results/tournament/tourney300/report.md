# Tournament Report — tourney300

**Games completed:** 100  
**Malformed skipped:** 0  
**Players per game:** 6

## Strategy Leaderboard

| Rank | Strategy | Games | Wins | Win% | 95% CI | Mean Placement | Betrayals | Alliances |
|------|----------|-------|------|------|--------|----------------|-----------|-----------|
| 1 | diplomat_coalition | 100 | 27 | 27.00% | [0.1927, 0.3643] | 2.99 | 197 | 231 |
| 2 | card_cycle_hunter | 100 | 25 | 25.00% | [0.1755, 0.3430] | 2.66 | 523 | 349 |
| 3 | aggressive_blitz | 100 | 19 | 19.00% | [0.1251, 0.2778] | 3.26 | 835 | 615 |
| 4 | australia_lock | 100 | 14 | 14.00% | [0.0853, 0.2214] | 3.48 | 456 | 313 |
| 5 | south_america_lock | 100 | 11 | 11.00% | [0.0625, 0.1863] | 3.74 | 422 | 375 |
| 6 | turtle_defensive | 100 | 4 | 4.00% | [0.0157, 0.0984] | 4.87 | 36 | 343 |

## Model Leaderboard

| Rank | Model | Games | Wins | Win% | 95% CI | Mean Placement |
|------|-------|-------|------|------|--------|----------------|
| 1 | gemma4:31b-cloud | 100 | 39 | 39.00% | [0.3002, 0.4880] | 2.48 |
| 2 | gemma4:cloud | 100 | 38 | 38.00% | [0.2910, 0.4779] | 2.19 |
| 3 | qwen3.5:cloud | 100 | 18 | 18.00% | [0.1170, 0.2667] | 2.83 |
| 4 | kimi-k2.6:cloud | 100 | 4 | 4.00% | [0.0157, 0.0984] | 3.86 |
| 5 | nemotron-3-super:cloud | 100 | 1 | 1.00% | [0.0018, 0.0545] | 4.69 |
| 6 | deepseek-v4-flash:cloud | 100 | 0 | 0.00% | [0.0000, 0.0370] | 4.95 |

## Strategy × Model Win Matrix

| Strategy | deepseek-v4-flash:cloud | gemma4:31b-cloud | gemma4:cloud | kimi-k2.6:cloud | nemotron-3-super:cloud | qwen3.5:cloud |
|---|---|---|---|---|---|---|
| aggressive_blitz | 0.0% (0/19) | 45.0% (9/20) | 27.8% (5/18) | 11.8% (2/17) | 0.0% (0/14) | 25.0% (3/12) |
| australia_lock | 0.0% (0/25) | 50.0% (9/18) | 26.7% (4/15) | 0.0% (0/15) | 0.0% (0/11) | 6.2% (1/16) |
| card_cycle_hunter | 0.0% (0/12) | 62.5% (10/16) | 58.3% (7/12) | 7.1% (1/14) | 0.0% (0/19) | 25.9% (7/27) |
| diplomat_coalition | 0.0% (0/18) | 61.5% (8/13) | 72.7% (16/22) | 5.9% (1/17) | 6.2% (1/16) | 7.1% (1/14) |
| south_america_lock | 0.0% (0/10) | 8.3% (1/12) | 21.1% (4/19) | 0.0% (0/15) | 0.0% (0/24) | 30.0% (6/20) |
| turtle_defensive | 0.0% (0/16) | 9.5% (2/21) | 14.3% (2/14) | 0.0% (0/22) | 0.0% (0/16) | 0.0% (0/11) |

## Survival & Placement

| Rank | Strategy | Mean Placement |
|------|----------|----------------|
| 1 | card_cycle_hunter | 2.66 |
| 2 | diplomat_coalition | 2.99 |
| 3 | aggressive_blitz | 3.26 |
| 4 | australia_lock | 3.48 |
| 5 | south_america_lock | 3.74 |
| 6 | turtle_defensive | 4.87 |

## Diplomacy (Alliances & Betrayals)

| Strategy | Betrayals | Alliances |
|----------|-----------|-----------|
| diplomat_coalition | 197 | 231 |
| card_cycle_hunter | 523 | 349 |
| aggressive_blitz | 835 | 615 |
| australia_lock | 456 | 313 |
| south_america_lock | 422 | 375 |
| turtle_defensive | 36 | 343 |
