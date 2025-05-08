using LitPOMDP

pomdp = LiPOMDP(
    demand=[1.0, 2.0],
    mine_output=1.2
)

s0 = State(
    deposits=[1.0],
    have_mined=[true]
)

###TODO: test initiation action and observation

a0=Action(
    site=0,
    type=1
)

o0=Observation(
    deposits=[10, 5, 60, 7],
    price=$10
)