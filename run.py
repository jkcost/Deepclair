import argparse
import pandas as pd
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *
from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv
import pdb
import setproctitle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import wandb

EVALUATION_METRICS = ['max_wealth', 'min_wealth', 'avg_wealth', 'final_wealth', 'ARR', 'ASR', 'AVOL', 'MDD', 'CR', 'DDR', 'long_ratio', 'short_ratio']


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)
    if func_args.method =='PG':
        func_args.msu_bool = False

    if func_args.market =='IXIC':
        func_args.stock = ['AAPL','MSFT','PEP','COST','ADBE','CSCO','CMCSA','TXN','INTC',
                         'QCOM','HON','AMGN','AMAT','ADP','ADI','VRTX','LRCX','REGN','MU','CDNS','CSX','KLAC',
           'MNST','CTAS','IDXX','PCAR']


    elif func_args.market =='DJIA':
        func_args.stock = ['UNH', 'HD', 'MCD', 'MSFT', 'AMGN', 'CAT', 'BA', 'HON', 'CVX', 'TRV', 'JNJ', 'AAPL','AXP', 'PG', 'WMT', 'JPM', 'IBM', 'NKE', 'MRK', 'MMM', 'DIS', 'KO', 'CSCO', 'VZ', 'WBA', 'INTC']

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H%M%S')

    wandb_obj = wandb.init(config=func_args, 
                           project=func_args.wandb_project_name, 
                           group=func_args.wandb_group_name,
                           name=func_args.wandb_session_name)
    wandb_obj.define_metric('batch_step')
    wandb_obj.define_metric('train/loss',       step_metric='batch_step')
    wandb_obj.define_metric('epoch_step')
    wandb_obj.define_metric('train/return',     step_metric='batch_step')
    wandb_obj.define_metric('train/avg_rho',    step_metric='batch_step')
    wandb_obj.define_metric('train/avg_mdd',    step_metric='batch_step')
    for split in ['train', 'valid', 'test']:
        for metric in EVALUATION_METRICS:
            wandb_obj.define_metric(f'{split}/{metric}', step_metric='epoch_step')
    WANDB_NAME = f'{func_args.wandb_project_name}_{func_args.wandb_group_name}_{func_args.wandb_session_name}'
    setproctitle.setproctitle(WANDB_NAME)


    if func_args.mode == 'train':
        PREFIX =  'outputs/'
        PREFIX = os.path.join(PREFIX,WANDB_NAME+'_'+start_time)
        PREFIX = PREFIX+'('+func_args.train_type+')' + '_'+func_args.exp_num+'_' +func_args.market+ '_'+func_args.model+'_' +'_r'+'('+str(func_args.r)+')'
        img_dir = os.path.join(PREFIX,'img_file')
        save_dir = os.path.join(PREFIX,'log_file')
        model_save_dir = os.path.join(PREFIX,'model_file')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        hyper = copy.deepcopy(func_args.__dict__)
        print(hyper)
        hyper['device'] = 'cuda' if hyper['device'] == torch.device('cuda') else 'cpu'
        json_str = json.dumps(hyper, indent=4)

        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)

        writer = SummaryWriter(save_dir)
        writer.add_text('hyper_setting', str(hyper))

        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('WARNING')

        fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fhlr.setFormatter(formatter)

        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        if func_args.market == 'DJIA':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            logger.debug(f'stocks_data.shape:{stocks_data.shape}')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            predict_data= pd.read_csv(data_prefix+'DJI_stocks.csv')
            predict_data['date'] = pd.to_datetime(predict_data['date'])
            index_data = pd.read_csv(data_prefix + '^DJI.csv')
            index_data['date'] = pd.to_datetime(index_data['date'])
            val_idx = index_data[index_data['date'] < func_args.valid_period].index[-1]
            test_idx = index_data[index_data['date'] <= func_args.test_period].index[-1]
            logger.warning(f"test date -> {index_data.iloc[test_idx]['Date']}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            allow_short = True

        elif func_args.market == 'IXIC':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            logger.debug(f'stocks_data.shape:{stocks_data.shape}')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            predict_data = pd.read_csv(data_prefix + 'IXIC_stocks.csv')
            predict_data['date'] = pd.to_datetime(predict_data['date'])
            index_data = pd.read_csv(data_prefix + '^IXIC.csv')
            index_data['date'] = pd.to_datetime(index_data['date'])
            val_idx = index_data[index_data['date'] < func_args.valid_period].index[-1]
            test_idx = index_data[index_data['date'] <= func_args.test_period].index[-1]
            logger.warning(f"test date -> {index_data.iloc[test_idx]['Date']}")
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            allow_short = True

        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,predict_data=predict_data,
                           in_features=func_args.in_features, val_idx=val_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size,fee=func_args.fee, window_len=func_args.window_len,
                           trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short,seq_len=func_args.seq_len,label_len=func_args.label_len,pred_len=func_args.pred_len)
        A = A[:rate_of_return.shape[0], :rate_of_return.shape[0]]
        supports = [A]
        actor = RLActor(supports, func_args).to(func_args.device)  # define type of network
        agent = RLAgent(env, actor, func_args)
        if func_args.train_type =='Lora' and func_args.method =='PRED_MSU':
            logger.warning(actor.res_dict)
        # wandb_obj.watch(actor, log='all', log_graph=True, log_freq=10)
        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        wandb_obj.watch(actor, log='all', log_graph=True, log_freq=10)


        logger.warning('Start training!!!')
        try:
            max_ARR = 0
            no_improve_epoch = 0
            MAX_NO_IMPROVE_EPOCH = 10
            for epoch in range(func_args.epochs):
                # Train Step
                epoch_return = 0
                epoch_loss = 0
                for j in range(mini_batch_num):
                    episode_return, avg_rho, avg_mdd, loss, rho_record_train = agent.train_episode()
                    epoch_return += episode_return
                    epoch_loss += loss
                    wandb_obj.log({'train/loss': loss, 
                                   'train/return': episode_return,
                                   'train/avg_rho': avg_rho, 
                                   'train/avg_mdd': avg_mdd, 
                                   'batch_step': j + epoch*mini_batch_num})
                    agent.scheduler.step()

                current_lr = agent.optimizer.param_groups[0]['lr']
                logger.warning(f"Epoch {epoch}, Current Learning Rate: {current_lr}")
                logger.warning(f'epoch({epoch}) loss -> {epoch_loss}')
                avg_train_return = epoch_return / mini_batch_num

                # Validation Step
                agent_wealth,rho_record,weight_record,future_p = agent.evaluation()
                metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
                for k, v in metrics.items():
                    if k in EVALUATION_METRICS:
                        if 'wealth' not in k:
                            wandb_obj.log({f'valid/{k}': v,                     'epoch_step': epoch})
                        if k == 'min_wealth':
                            wandb_obj.log({f'valid/{k}': np.min(agent_wealth),  'epoch_step': epoch})
                        if k == 'max_wealth':
                            wandb_obj.log({f'valid/{k}': np.max(agent_wealth),  'epoch_step': epoch})
                        if k == 'avg_wealth':
                            wandb_obj.log({f'valid/{k}': np.mean(agent_wealth), 'epoch_step': epoch})
                        if k == 'final_wealth':
                            wandb_obj.log({f'valid/{k}': agent_wealth[-1, -1],  'epoch_step': epoch})

                if metrics['ARR'] > max_ARR :
                    logger.warning(f'New Best ARR Policy in {epoch}!!!!')
                    no_improve_epoch = 0
                    invest_record = json.dumps(weight_record)
                    max_ARR = metrics['ARR']
                    save_dict = {'model_state_dict': actor.state_dict(), 'epoch': epoch, 'loss': epoch_loss}
                    record_path = os.path.join(save_dir,'invest_record.json')
                    with open(record_path, 'w') as f:
                        json.dump(invest_record, f, indent=4)
                    torch.save(save_dict, os.path.join(model_save_dir, 'best_arr.pkl'))
                else:
                    no_improve_epoch +=1
                if no_improve_epoch >= MAX_NO_IMPROVE_EPOCH and epoch < 50:
                    logger.warning('!!!!!!!!!!!!!!!policy reset !!!!!!!!')
                    agent.actor = RLActor(supports, func_args).to(func_args.device)

        except KeyboardInterrupt:
            logger.warning('End of training...????')
            save_dict = {'model_state_dict': actor.state_dict(), 'epoch': epoch, 'loss': epoch_loss}
            torch.save(save_dict, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))

        #for test
        actor = RLActor(supports, func_args).to(func_args.device)  # define type of network
        actor.load_state_dict(torch.load(os.path.join(model_save_dir, 'best_arr.pkl')))
        logger.warning("Successfully loaded best checkpoint...")
        agent = RLAgent(env, actor, func_args)
        agent_wealth, rho_record, weight_record, future_p = agent.evaluation('test')
        metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
        for k, v in metrics.items():
            if k in EVALUATION_METRICS:
                if 'wealth' not in k:
                    wandb_obj.log({f'test/{k}': v, 'epoch_step': epoch})
                if k == 'min_wealth':
                    wandb_obj.log({f'test/{k}': np.min(agent_wealth), 'epoch_step': epoch})
                if k == 'max_wealth':
                    wandb_obj.log({f'test/{k}': np.max(agent_wealth),  'epoch_step': epoch})
                if k == 'avg_wealth':
                    wandb_obj.log({f'test/{k}': np.mean(agent_wealth), 'epoch_step': epoch})
                if k == 'final_wealth':
                    wandb_obj.log({f'test/{k}': agent_wealth[-1, -1],  'epoch_step': epoch})
        np.save(img_dir + f'/last_agent_wealth_{func_args.exp_num}.npy', agent_wealth)
        np.save(img_dir+f'/last_rho_record.npy_{func_args.exp_num}',np.array(rho_record))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project_name', '-wpn', type=str, default='CIKM2024-DeepClair')
    parser.add_argument('--wandb_group_name',   '-wgn', type=str, default='Debugging Mode')
    parser.add_argument('--wandb_session_name', '-wsn', type=str, default='Debugging Mode')

    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--method', type=str, default='PRED_MSU')
    parser.add_argument('--model',type=str,default='Fedformer')
    parser.add_argument('--market', type=str, default='IXIC')
    parser.add_argument('--train_type', type=str, default='Lora',
                        help='model name, options: [Lora,Frozen,Fine]')
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--valid_period',type=str,default='2004-10-15')
    parser.add_argument('--test_period', type=str,default='2006-06-01')
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--exp_num', type=str, default='1')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false', default=False)
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false', default=False)
    parser.add_argument('--stocks', type=str, nargs='+',
                        default=['UNH', 'HD', 'MCD', 'MSFT', 'AMGN', 'CAT', 'BA', 'HON', 'CVX', 'TRV', 'JNJ', 'AAPL',
                                 'AXP', 'PG', 'WMT', 'JPM', 'IBM', 'NKE', 'MRK', 'MMM', 'DIS', 'KO', 'CSCO', 'VZ',
                                 'WBA', 'INTC'],
                        help='set of stocks')


    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    #forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=5, help='start token length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'


    # model define
    parser.add_argument('--enc_in', type=int, default=26, help='encoder input size')  # number of input feature
    parser.add_argument('--dec_in', type=int, default=26, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=26, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)


