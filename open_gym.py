import gym

def mujoco_make(render, name):
    import mujoco
    if "pusher" in name:
        return mujoco.Pusher(render)
    return mujoco.Reacher(render)

def ergojr_make(render, name):
    if "pusher" in name:
        from ergo_pusher_env import ErgoPusherEnv
        return ErgoPusherEnv(not render)
    else:
        from ergo_reacher_env import ErgoReacherEnv
        backlash = "backlash" in name
        if "simple" in name:
            return ErgoReacherEnv(not render, goals=1, multi_goal=False, terminates=False, simple=True, gripper=False, backlash=backlash)
        else:
            return ErgoReacherEnv(not render, goals=1, multi_goal=False, terminates=False, simple=False, gripper=False, backlash=backlash)

def panda_make(render, name):
    import panda
    if "pusher" in name:
        return panda.Pusher(render)
    return panda.Reacher(render)

def make_env(env_name, render, colab=True):
    render = render and not colab
    if "panda" in env_name:
        return panda_make(render, env_name)
    elif "ergojr" in env_name:
        return ergojr_make(render, env_name)
    elif "mujoco" in env_name:
        return mujoco_make(render, env_name)
    assert False

