import config

import torch
import torch.nn as nn

from normalizer import Normalizer

class RunningNorm(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.add_module("norm", Normalizer(size_in))

        self.stop = False
        for p in self.norm.parameters():
            p.requires_grad = False

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def stop_norm(self):
        self.stop = True

    def active(self):
        return not self.stop

    def forward(self, states):
        shape = states.shape
        states = states.to(self.device()).view(-1, self.norm.size)
        if not self.stop:
            self.norm.update(states)
        return self.norm.normalize(states).view(shape)

goal_norm = RunningNorm(config.GOAL_SIZE)
goal_norm.share_memory()

class GlobalNormalizerWithTime(nn.Module):
    def __init__(self, size_in, lowlevel=True):
        super().__init__()

        self.lowlevel = lowlevel
        
        enc = RunningNorm((size_in if config.LEAK2LL or not lowlevel else config.LL_STATE_SIZE) - config.TIMEFEAT)
        
        self.add_module("enc", enc)
        self.add_module("goal_encoder", goal_norm)

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def stop_norm(self):
        self.enc.stop_norm()

    def active(self):
        return self.enc.active()

    def forward(self, states):
        states = states.to(self.device())

        if config.TIMEFEAT:
            tf = states[:, -config.TIMEFEAT:].view(-1, 1)
            states = states[:, :-config.TIMEFEAT]

        #print("\n my device", self.device(), self.goal_encoder.device(), goal_norm.device())
        enc = lambda data: self.goal_encoder(data).view(len(data), -1)
        pos = lambda buf, b, e: buf[:, b*config.GOAL_SIZE:e*config.GOAL_SIZE]

        # goal, {current, previous} arm pos 
        arm_pos = pos(states, 1, 3)
        if config.LEAK2LL or not self.lowlevel:# achieved goal and actual goal leaking hint what is our goal to low level
            arm_pos = enc(
                    torch.cat([pos(states, 0, 1), arm_pos, pos(states, -1, 10000)], 1)
                    )[:, :-config.GOAL_SIZE] # skip goal from here, just add hint via norm, also achieved one will be not used by NN
        else:
            arm_pos = torch.cat([pos(arm_pos, 0, 1),
                enc( arm_pos )
                ], 1)# first arm_pos aka fake achieved will be not used anyway

        obj_pos_w_goal = pos(states, -(2*config.PUSHER+1), 10000)
        if config.PUSHER and not self.lowlevel: # object pos, prev pos, goal --> only high level
            obj_pos_w_goal = enc(obj_pos_w_goal)

        state = states
        if self.lowlevel and not config.LEAK2LL:
            state = states[:, config.GOAL_SIZE:][:, :config.LL_STATE_SIZE-config.TIMEFEAT]

        encoded = self.enc(state)

        if self.lowlevel and not config.LEAK2LL:
            encoded = torch.cat([states[:, :config.GOAL_SIZE], encoded, states[:, config.GOAL_SIZE+config.LL_STATE_SIZE-config.TIMEFEAT:]], 1)

        encoded = pos(encoded, 3, -(2*config.PUSHER+1)) # skip object + goal positions, those will be added by goal_encoder

        # note : achievedel goal, and goal will be skipped from NN ( this norm is used in encoders just for puting it trough NN )
        #print("\n sumarize", arm_pos.device, encoded.device, obj_pos_w_goal.device)
        encoded = torch.cat([arm_pos, encoded, obj_pos_w_goal], 1)

        if states.shape[-1] != encoded.shape[-1]:
            print("\nshappezzz:", states.shape[-1], encoded.shape[-1], self.lowlevel, obj_pos_w_goal.shape)
        assert states.shape[-1] == encoded.shape[-1]
        if config.TIMEFEAT:
            encoded = torch.cat([encoded, tf], 1)
        return encoded
