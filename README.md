# LitPOMDP

[![Build Status](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Version 2.0 - Multi-Site Action Space with Automatic Production

### Major Changes
This version introduces a significant restructuring of the action space to support simultaneous actions across multiple mining sites with automatic production. Key changes include:

#### 1. **Action Space Redesign**
- **Old**: Single site action per timestep (`Action(site=Int, type=Int)`)
- **New**: Vector-based actions for all sites (`Action(site=Vector{Int}, type=Vector{Int})`)
- Allows simultaneous actions across all 4 mining sites
- Action types: 0=DoNothing, 1=Explore, 2=Open
- **Removed**: Operate action (type 3) - mines now produce automatically after opening
- **Action space size**: 3^4 = 81 possible action combinations

#### 2. **Simplified State Representation**
- **Removed**: `have_operated` and `ready_to_operate` boolean vectors
- **Consolidated**: Single `have_opened` boolean vector tracking mine status
- **Automatic Production**: Opened mines produce lithium automatically each year

#### 3. **Reward Function Updates**
- Automatic production calculation from all opened mines
- No action needed to operate mines - they produce if opened and have remaining deposits
- Aggregates costs (CAPEX for opening, OPEX for all producing mines)
- Emissions calculated for all producing mines
- Market revenue based on total sellable production (limited by company demand)
- Exploration costs (1000 per site) when exploring

#### 4. **Transition Logic**
- Opening actions set `have_opened[site] = true`
- All opened mines automatically produce each timestep
- Production randomness (±10%) applied independently to each producing mine
- Deposits decrease and volume increases automatically for opened mines

#### 5. **Complete Action Space**
The action space now includes all 81 combinations:
- Each site can independently: DoNothing (0), Explore (1), or Open (2)
- Examples: 
  - `[0,0,0,0]` - Do nothing at all sites
  - `[2,0,0,0]` - Open site 1 only
  - `[2,2,2,2]` - Open all sites simultaneously
  - `[1,2,0,0]` - Explore site 1 and open site 2

#### 6. **Belief Updates**
- Tracks mine openings through action observations
- Updates cumulative volume based on automatic production from all opened mines
- Simplified belief structure without operational status tracking


## Dev Notes

- [x] Added Codespace and bethpage
- [x] Module installation and dependencies
- [x] Fix the `observation` function
    - [x] We also add `gen` function
- [ ] Fix the belief update (Mansur)
- [ ] Go over `initialstate` function (Mansur)
- [ ] Add `hist_data` to `LiPOMDP` to be able to use historical data (Anna)
- [ ] Add slope + intercept and exponential parameters for pricing models
- [ ] Set up the `LiPOMDP` to be able to 
    - [ ] use different pricing models 
        - [ ] historical model, 
        - [ ] simple model (static, exponential, linear), 
        - [ ] dynamic model (action- and state-dependent))
    - [ ] add stochastic option
- [ ] Run experiments (plan vs. eval)

