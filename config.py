ENV = "panda-reach"
SEED = 0
EVAL_FREQ = 5e3
MAX_TIMESTEPS = 1e6
# TD3
EXPL_NOISE = 0.1
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
# TD3 + BC
ALPHA = 2.5
NORMALIZE = True
# OPEN AI TD3 BASELINE TRAINING
STEPS_PER_EPOCH = 4000
EPOCHS =100
REPLAY_SIZE = 1e6
START_STEPS = 10000
UPDATE_AFTER = 1000
UPDATE_EVERY = 50
# HER
HER_PER_EP = 20
HER_RATIO = 1.

#DLPPOH
TIMEFEAT = False#True#
LEAK2LL = True#False#

PANDA = "panda" in ENV
ERGOJR = "ergojr" in ENV
MUJOCO = not PANDA and not ERGOJR
assert MUJOCO + PANDA + ERGOJR == 1

BACKLASH = False
PUSHER = "usher" in ENV

GOAL_SIZE = 3

PUSHER = False#True#

if ERGOJR: # no gripper, velo per joint ( #of joints == action_size )
    ACTION_SIZE = 3 + (not PUSHER) * 1#3
    LL_STATE_SIZE = GOAL_SIZE * 2 + ACTION_SIZE * 2 + TIMEFEAT
    STATE_SIZE = GOAL_SIZE + LL_STATE_SIZE + 3*GOAL_SIZE*PUSHER
else: # arm pos, arm prev pos, arm velo, gripper pos + velo + velp
    ACTION_SIZE = 3 + MUJOCO
    LL_STATE_SIZE = GOAL_SIZE * 3 + 4 * MUJOCO + TIMEFEAT
    STATE_SIZE = GOAL_SIZE + LL_STATE_SIZE + 6*GOAL_SIZE*PUSHER# velp + gripper, object velp for pusher
