import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from model.ASU import ASU
from model.MSU import MSU
from model.EIIE import EIIE
from model.FED_PRED import FED_PORT
from model.AUTO_PRED import AUTO_PORT
from model.INFO_PRED import INFO_PORT
from model.TRANS_PRED import TRANS_PORT

from torch.optim.lr_scheduler import CyclicLR
import torch.nn.init as init

from peft import LoraConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
EPS = 1e-20

import pdb


def replace_nan_with_zero_and_report(grad, param_name, param_value):
    if torch.isnan(grad).any():
        print(f"Parameter {param_name} with value {param_value.detach()} has NaN gradient!")
        grad[torch.isnan(grad)] = 0
    return grad


def register_nan_hooks_to_model(model):
    """
    Register hooks to model's parameters. If a gradient contains NaN values,
    they will be replaced with zeros and the parameter's name and value will be reported.
    """
    handles = []
    for name, param in model.named_parameters():
        if param.requires_grad:  # Check if the parameter requires gradient
            handle = param.register_hook(lambda grad, name=name, param=param: replace_nan_with_zero_and_report(grad, name, param))
            handles.append(handle)
    return handles


def lora_get_trainable_parameters_result(lora_module):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in lora_module.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    res_dict =  {
            "trainable params": f"{trainable_params:,d}",
            "all params": f"{all_param:,d}",
            "trainable%": f"{100 * trainable_params / all_param}",
            "print": f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"

            }

    return res_dict



class RLActor(nn.Module):
    def __init__(self, supports, args):
        super(RLActor, self).__init__()
        if args.method =='PG':
            self.EIIE = EIIE(batch_size=args.batch_size,
                             num_stock=args.num_assets,
                             in_features=args.in_features[0],
                             window_len=args.window_len)
        elif args.method == 'PRED':
            if args.model =='Fedformer':
                self.pred_port = FED_PORT(args)
            elif args.model =='Autoformer':
                self.pred_port = AUTO_PORT(args)
            elif args.model == 'Informer':
                self.pred_port = INFO_PORT(args)
            elif args.model == 'Transformer':
                self.pred_port = TRANS_PORT(args)
            self.pred_port.load_state_dict(torch.load(f'./model/checkpoint_{args.model}_{args.pred_len}_{args.market}.pth'))
            # self.pred_port.eval()
            self.asu = ASU(num_nodes=args.num_assets,
                           in_features=args.in_features[0],
                           hidden_dim=args.hidden_dim,
                           window_len=args.window_len,
                           dropout=args.dropout,
                           kernel_size=args.kernel_size,
                           layers=args.num_blocks,
                           supports=supports,
                           spatial_bool=args.spatial_bool,
                           addaptiveadj=args.addaptiveadj)

            if args.msu_bool:
                self.msu = MSU(in_features=args.in_features[1],
                               window_len=args.window_len,
                               hidden_dim=args.hidden_dim)

        elif args.method =='PRED_MSU':
            self.asu = ASU(num_nodes=args.num_assets,
                           in_features=args.in_features[0],
                           hidden_dim=args.hidden_dim,
                           window_len=args.window_len,
                           dropout=args.dropout,
                           kernel_size=args.kernel_size,
                           layers=args.num_blocks,
                           supports=supports,
                           spatial_bool=args.spatial_bool,
                           addaptiveadj=args.addaptiveadj)


            self.msu_linear = nn.Linear(args.window_len * args.num_assets, 2)
            init.xavier_uniform_(self.msu_linear.weight)


            if args.model =='Fedformer':
                self.msu = FED_PORT(args)
            elif args.model =='Autoformer':
                self.msu = AUTO_PORT(args)
            elif args.model == 'Informer':
                self.msu = INFO_PORT(args)
            elif args.model == 'Transformer':
                self.msu = TRANS_PORT(args)
            self.msu.load_state_dict(torch.load(f'./model/checkpoint_{args.model}_{args.pred_len}_{args.market}.pth'))

            if args.train_type =='Lora':
                lora_regex = r'encoder.attn_layers.\d.attention\.[A-Za-z]+_projection'
                self.lora_config = LoraConfig(target_modules=lora_regex, lora_dropout=0.1,bias='lora_only',r=args.r)
                self.msu = get_peft_model(self.msu,self.lora_config)
                self.msu.print_trainable_parameters()
                self.res_dict = lora_get_trainable_parameters_result(self.msu)

        else:
            self.asu = ASU(num_nodes=args.num_assets,
                           in_features=args.in_features[0],
                           hidden_dim=args.hidden_dim,
                           window_len=args.window_len,
                           dropout=args.dropout,
                           kernel_size=args.kernel_size,
                           layers=args.num_blocks,
                           supports=supports,
                           spatial_bool=args.spatial_bool,
                           addaptiveadj=args.addaptiveadj)

            if args.msu_bool:
                self.msu = MSU(in_features=args.in_features[1],
                               window_len=args.window_len,
                               hidden_dim=args.hidden_dim)
        self.args = args


    def forward(self, x_a, x_m,x_p, masks=None, deterministic=False, logger=None, y=None):
        if self.args.method=='PG':
            scores = self.EIIE(x_a)
            res = None
        elif self.args.method=='PRED':
            batch_x = torch.from_numpy(x_p[0]).float().to(self.args.device)
            batch_x_mark = torch.from_numpy(x_p[2]).float().to(self.args.device)
            batch_y_mark = torch.from_numpy(x_p[3]).float().to(self.args.device)
            # decoder input
            batch_y = torch.from_numpy(x_p[1]).float().to(self.args.device)
            dec_inp = torch.zeros_like(batch_y[:, - self.args.trade_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            pred_scores = self.pred_port(batch_x,batch_x_mark,dec_inp,batch_y_mark)
            pred_scores = pred_scores.transpose(1,2)

            combined_x = torch.cat((x_a, pred_scores.unsqueeze(-1).expand(-1, -1, -1, x_a.shape[-1])), dim=2)

            scores = self.asu(combined_x,masks)
            if self.args.msu_bool:
                res = self.msu(x_m)
            else:
                res = None
        elif self.args.method=='PRED_MSU':
            batch_x = torch.from_numpy(x_p[0]).float().to(self.args.device)
            batch_x_mark = torch.from_numpy(x_p[2]).float().to(self.args.device)
            batch_y_mark = torch.from_numpy(x_p[3]).float().to(self.args.device)
            batch_y = torch.from_numpy(x_p[1]).float().to(self.args.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, - self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.device)

            pred_scores = self.msu(batch_x, batch_x_mark, dec_inp, batch_y_mark).detach() #[batch,pred_len,stock_num]


            x = pred_scores.view(pred_scores.shape[0],-1)
            activation =  nn.LeakyReLU(negative_slope=0.01)
            x = activation(x)
            pred_rho = self.msu_linear(x)

            scores = self.asu(x_a, masks)
            res = pred_rho



        else:
            scores = self.asu(x_a, masks)
            if self.args.msu_bool:
                res = self.msu(x_m)
            else:
                res = None
        return self.__generator(scores, res, deterministic)

    def __generator(self, scores, res, deterministic=None):
        weights = np.zeros((scores.shape[0], 2 * scores.shape[1]))

        winner_scores = scores
        loser_scores = scores.sign() * (1 - scores)

        scores_p = torch.softmax(scores, dim=-1)

        w_s, w_idx = torch.topk(winner_scores.detach(), self.args.G)

        long_ratio = torch.softmax(w_s, dim=-1)

        for i, indice in enumerate(w_idx):
            weights[i, indice.detach().cpu().numpy()] = long_ratio[i].cpu().numpy()

        l_s, l_idx = torch.topk(loser_scores.detach(), self.args.G)

        short_ratio = torch.softmax(l_s.detach(), dim=-1)
        for i, indice in enumerate(l_idx):
            weights[i, indice.detach().cpu().numpy() + scores.shape[1]] = short_ratio[i].cpu().numpy()
        if self.args.method == 'PRED_MSU':
            rho = res
            rho_log_p = None
            mu = res[...,0]
            activation= nn.Tanh()
            mu = activation(mu)
            sigma = torch.log(1 + torch.exp(res[...,1]))
            if deterministic:
                rho = torch.clamp(mu, 0.0, 1.0)
                rho_log_p = None
            else:
                try:
                    m = Normal(mu, sigma)
                    sample_rho = m.sample()
                    rho = torch.clamp(sample_rho, 0.0, 1.0)
                    rho_log_p = m.log_prob(sample_rho)
                except:
                    print()
        else:
            if self.args.msu_bool:
                mu = res[..., 0]
                sigma = torch.log(1 + torch.exp(res[..., 1]))
                if deterministic:
                    rho = torch.clamp(mu, 0.0, 1.0)
                    rho_log_p = None
                else:
                    m = Normal(mu, sigma)
                    sample_rho = m.sample()
                    rho = torch.clamp(sample_rho, 0.0, 1.0)
                    rho_log_p = m.log_prob(sample_rho)
            else:
                rho = torch.ones((weights.shape[0])).to(self.args.device) * 0.5
                rho_log_p = None
        return weights, rho, scores_p, rho_log_p


class RLAgent():
    def __init__(self, env, actor, args, logger=None):
        self.actor = actor
        self.env = env
        self.args = args
        self.logger = logger

        self.total_steps = 0
        # Filter out self.msu parameters
        if args.train_type == 'Frozen':
            params_to_optimize = [param for name, param in self.actor.named_parameters() if "msu" not in name]
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.scheduler = CyclicLR(self.optimizer, base_lr=self.args.lr ,max_lr=self.args.lr*100, cycle_momentum=False)

    def train_episode(self):

        self.__set_train()

        states, masks = self.env.reset()
        if self.args.method =='PG':
            self.actor.EIIE.reset(states[0])

        steps = 0
        batch_size = states[0].shape[0]

        steps_log_p_rho = []
        steps_reward_total = []
        steps_asu_grad = []

        rho_records = []

        agent_wealth = np.ones((batch_size, 1), dtype=np.float32)

        while True:
            steps += 1

            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.method == 'PRED' or self.args.method == 'PRED_MSU':
                x_p = states[2]

            else:
                x_p = None

            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None

            weights, rho, scores_p, log_p_rho \
                = self.actor(x_a, x_m,x_p, masks, deterministic=False)

            ror = torch.from_numpy(self.env.ror).to(self.args.device)
            normed_ror = (ror - torch.mean(ror, dim=-1, keepdim=True)) / \
                         torch.std(ror, dim=-1, keepdim=True)

            next_states, rewards, rho_labels, masks, done, info = \
                self.env.step(weights, rho.detach().cpu().numpy())

            steps_log_p_rho.append(log_p_rho)
            steps_reward_total.append(rewards.total - info['market_avg_return']) #over market_avg_return

            asu_grad = torch.sum(normed_ror * scores_p, dim=-1)
            ###advise
            steps_asu_grad.append(asu_grad)


            agent_wealth = np.concatenate((agent_wealth, info['total_value'][..., None]), axis=1)
            states = next_states

            rho_records.append(np.mean(rho.detach().cpu().numpy()))

            if done:
                if self.args.method =='PRED_MSU':
                    steps_log_p_rho = torch.stack(steps_log_p_rho, dim=-1)
                else:
                    if self.args.msu_bool:
                        steps_log_p_rho = torch.stack(steps_log_p_rho, dim=-1)

                steps_reward_total = np.array(steps_reward_total).transpose((1, 0))

                rewards_total = torch.from_numpy(steps_reward_total).to(self.args.device)
                mdd = self.cal_MDD(agent_wealth)

                rewards_mdd = - 2 * torch.from_numpy(mdd - 0.5).to(self.args.device)

                rewards_total = (rewards_total - torch.mean(rewards_total, dim=-1, keepdim=True)) \
                                / torch.std(rewards_total, dim=-1, keepdim=True)

                gradient_asu = torch.stack(steps_asu_grad, dim=1)

                if self.args.method =='PRED_MSU':
                    gradient_rho = (rewards_total * steps_log_p_rho)  # Return_version
                    loss = - (self.args.gamma * gradient_rho + gradient_asu)
                else:
                    if self.args.msu_bool:
                        # gradient_rho = (rewards_mdd * steps_log_p_rho) # MDD version
                        gradient_rho = (rewards_total * steps_log_p_rho) #Return_version
                        loss = - (self.args.gamma * gradient_rho + gradient_asu)
                    else:
                        loss = - (gradient_asu)
                loss = loss.mean()
                assert not torch.isnan(loss)
                self.optimizer.zero_grad()
                loss = loss.contiguous()
                loss.backward()
                self.optimizer.step()
                break

        rtns = (agent_wealth[:, -1] / agent_wealth[:, 0]).mean()
        avg_rho = np.mean(rho_records)
        avg_mdd = mdd.mean()
        return rtns, avg_rho, avg_mdd,loss,rho_records





    def evaluation(self, logger=None):
        if logger =='test':
            self.__set_test()
        else:
            self.__set_eval()
        states, masks = self.env.reset()
        if self.args.method == 'PG':
            self.actor.EIIE.reset(states[0])
        steps = 0
        batch_size = states[0].shape[0]

        agent_wealth = np.ones((batch_size, 1), dtype=np.float32)
        rho_record = []
        weights_record = {'stock': [],
                          'long_ratio': [],
                          'short_ratio': [],
                          'stock_return': [],
                          'avg_return': []}

        future_p_record = []
        while True:
            steps += 1
            x_a = torch.from_numpy(states[0]).to(self.args.device)
            masks = torch.from_numpy(masks).to(self.args.device)
            if self.args.method == 'PRED' or self.args.method =='PRED_MSU':
                x_p = states[2]
            else:
                x_p = None

            if self.args.msu_bool:
                x_m = torch.from_numpy(states[1]).to(self.args.device)
            else:
                x_m = None

            weights, rho, _, _\
                = self.actor(x_a, x_m,x_p, masks, deterministic=True)
            next_states, rewards, future_p, masks, done, info = self.env.step(weights, rho.detach().cpu().numpy())

            agent_wealth = np.concatenate((agent_wealth, info['total_value'][..., None]), axis=-1)
            rho_record.append(info['p'][0].tolist())
            non_zero_indices_long = np.nonzero(weights)[1][:self.args.G]
            non_zero_indices_short = np.nonzero(weights)[1][self.args.G:]
            non_zero_values_long = weights[:,non_zero_indices_long][0]
            non_zero_values_short = weights[:,non_zero_indices_short][0]
            stock_lst = [self.args.stocks[i] for i in non_zero_indices_long.tolist()]
            return_lst = [info['market_fluctuation'][0][i] for i in non_zero_indices_long.tolist()]
            weights_record['stock'].append(stock_lst +return_lst)
            future_p_record.append(future_p)
            weights_record['long_ratio'].append(non_zero_values_long.tolist())
            weights_record['short_ratio'].append(non_zero_values_short.tolist())
            weights_record['stock_return'].append(info['market_fluctuation'][0].tolist())
            weights_record['stock_return'].append(info['market_avg_return'][0].tolist())
            states = next_states

            if done:
                break

        return agent_wealth,rho_record,weights_record,future_p_record


    def __set_train(self):
        self.actor.train()
        if self.args.method =='PRED_MSU' and self.args.train_type=='Frozen':
            self.actor.msu.eval()
        self.env.set_train()

    def __set_eval(self):
        self.actor.eval()
        self.env.set_eval()

    def __set_test(self):
        self.actor.eval()
        self.env.set_test()

    def cal_MDD(self, agent_wealth):
        drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) / \
                   np.maximum.accumulate(agent_wealth, axis=-1)
        MDD = np.max(drawdown, axis=-1)
        return MDD[..., None].astype(np.float32)

    def cal_CR(self, agent_wealth):
        pr = np.mean(agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1, axis=-1, keepdims=True)
        mdd = self.cal_MDD(agent_wealth)
        softplus_mdd = np.log(1 + np.exp(mdd))
        CR = pr / softplus_mdd
        return CR
