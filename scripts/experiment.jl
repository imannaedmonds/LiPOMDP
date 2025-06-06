using LitPOMDP
using POMDPs
using Random


pomdp = LiPOMDP(
    demand=[1.0, 2.0],
    mine_output=1.2,
    price_model_type = 3,
)
rng = MersenneTwister(1234)

s0 = State(
    deposits=[1.0],
    have_mined=[true],
    mine_ready=[false]
)

###TODO: test initiation action and observation

a0=Action(
    site=0,
    type=1
)

o0=Observation(
    deposits=[10, 5, 60, 7],
    price=10
)

prob_s1=transition(pomdp, s0, a0)
s1 = rand(rng, prob_s1)

r = reward(pomdp, s0, a0)
