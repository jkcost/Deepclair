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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import wandb





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

    wandb_obj = wandb.init(config=func_args, project='DeepClairDebug', name='dev_mode')

    wandb_obj.define_metric('batch_step')

    wandb_obj.define_metric('train/loss', step_metric='batch_step')

    wandb_obj.define_metric('epoch_step')

    wandb_obj.define_metric('train/avg_return', step_metric='epoch_step')

    wandb_obj.define_metric('train/avg_rho', step_metric='epoch_step')

    wandb_obj.define_metric('train/avg_mdd', step_metric='epoch_step')

    wandb_obj.define_metric('test/apr', step_metric='epoch_step')

    wandb_obj.define_metric('test/mdd', step_metric='epoch_step')

    wandb_obj.define_metric('test/avol', step_metric='epoch_step')

    wandb_obj.define_metric('test/asr', step_metric='epoch_step')

    wandb_obj.define_metric('test/sor', step_metric='epoch_step')

    wandb_obj.define_metric('test/cr', step_metric='epoch_step')

    wandb_obj.define_metric('test/min_wealth', step_metric='epoch_step')

    wandb_obj.define_metric('test/max_wealth', step_metric='epoch_step')

    wandb_obj.define_metric('test/avg_wealth', step_metric='epoch_step')

    wandb_obj.define_metric('test/final_wealth', step_metric='epoch_step')

    wandb_obj.define_metric('test/stock_lst', step_metric='epoch_step')

    wandb_obj.define_metric('test/long_ratio', step_metric='epoch_step')

    wandb_obj.define_metric('test/short_ratio', step_metric='epoch_step')

    wandb_obj.define_metric('test/trade_len_stock_return', step_metric='epoch_step')

    if func_args.mode == 'train':
        PREFIX =  'outpus/'
        PREFIX = os.path.join(PREFIX,start_time)
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


        try:
            max_ARR = 0
            no_improve_epoch = 0
            MAX_NO_IMPROVE_EPOCH = 10
            for epoch in range(func_args.epochs):
                epoch_return = 0
                epoch_loss = 0
                # for j in tqdm(range(mini_batch_num)):
                for j in range(mini_batch_num):
                    if j == 107:
                        print()

                    episode_return, avg_rho, avg_mdd,loss,rho_record_train = agent.train_episode()
                    epoch_return += episode_return
                    epoch_loss += loss
                    wandb_obj.log({'train/loss': epoch_loss})
                    agent.scheduler.step()



                current_lr = agent.optimizer.param_groups[0]['lr']
                logger.warning(f"Epoch {epoch}, Current Learning Rate: {current_lr}")
                logger.warning(f'epoch({epoch}) loss -> {epoch_loss}')
                avg_train_return = epoch_return / mini_batch_num
                logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f' %
                               (start_time, epoch, avg_train_return, avg_rho, avg_mdd))
                agent_wealth,rho_record,weight_record,future_p = agent.evaluation()
                metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
                writer.add_scalar('Test/ARR', metrics['ARR'], global_step=epoch)
                writer.add_scalar('Test/MDD', metrics['MDD'], global_step=epoch)
                writer.add_scalar('Test/AVOL', metrics['AVOL'], global_step=epoch)
                writer.add_scalar('Test/ASR', metrics['ASR'], global_step=epoch)
                writer.add_scalar('Test/SoR', metrics['DDR'], global_step=epoch)
                writer.add_scalar('Test/CR', metrics['CR'], global_step=epoch)
                logger.warning('after training %d round, max wealth: %.4f, min wealth: %.4f,'
                              ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                              'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                              % (
                                  epoch, max(agent_wealth[0]), min(agent_wealth[0]), np.mean(agent_wealth),
                                  agent_wealth[-1, -1], 100 * metrics['ARR'], metrics['ASR'], metrics['AVOL'],
                                  100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                              ))
                if metrics['ARR'] > max_ARR :
                    logger.warning(f'New Best ARR Policy in {epoch}!!!!')
                    logger.warning(f'rho_record_train:{rho_record_train[:500]}')
                    logger.warning(f'rho_record:{rho_record}')
                    no_improve_epoch = 0
                    invest_record = json.dumps(weight_record)

                    max_ARR = metrics['ARR']

                    record_path = os.path.join(save_dir,'invest_record.json')
                    with open(record_path, 'w') as f:
                        json.dump(invest_record, f, indent=4)
                    torch.save(actor, os.path.join(model_save_dir, 'best_cr.pkl'))

                else:
                    no_improve_epoch +=1


                if no_improve_epoch >= MAX_NO_IMPROVE_EPOCH and epoch < 50:
                    logger.warning('!!!!!!!!!!!!!!!policy reset !!!!!!!!')
                    agent.actor = RLActor(supports, func_args).to(func_args.device)

            test_record = {'max_wealth' :[],
                           'min_wealth' :[],
                           'avg_wealth' :[],
                           'final_wealth':[],
                           'ARR':[],
                           'ASR':[],
                           'AVOL':[],
                           'MDD':[],
                           'CR':[],
                           'DDR':[]}
            test_record['max_wealth'].append(max(agent_wealth[0]))
            test_record['min_wealth'].append(min(agent_wealth[0]))
            test_record['avg_wealth'].append(np.mean(agent_wealth[0]))
            test_record['final_wealth'].append((agent_wealth[-1, -1]))
            test_record['ARR'].append(100 * metrics['ARR'])
            test_record['ASR'].append(metrics['ASR'])
            test_record['AVOL'].append(metrics['AVOL'])
            test_record['MDD'].append(100 * metrics['MDD'])
            test_record['CR'].append(metrics['CR'])
            test_record['DDR'].append(metrics['DDR'])


            close_prices = index_data[index_data['date'] >= func_args.test_period]['Adj Close'].values
            index_agent_wealth = np.cumprod(1 + np.diff(close_prices) / close_prices[:-1])
            index_agent_wealth = np.insert(index_agent_wealth, 0, 1)
            index_agent_wealth = index_agent_wealth[np.newaxis, :]
            index_metrics = calculate_metrics(index_agent_wealth,trade_mode='D')




            logger.warning('test record : max wealth: %.4f, min wealth: %.4f,'
                           ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                           'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                           % (
                             np.mean(test_record['max_wealth']), np.mean(test_record['min_wealth']),
                               np.mean(test_record['avg_wealth']),
                               np.mean(test_record['final_wealth']), np.mean(test_record['ARR']),np.mean(test_record['ASR']), np.mean(test_record['AVOL']),
                               np.mean(test_record['MDD']),np.mean(test_record['CR']), np.mean(test_record['DDR'])
                           ))




            logger.warning('index_record:  max wealth: %.4f, min wealth: %.4f,'
                           ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                           'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                           % (
                                max(index_agent_wealth[0]), min(index_agent_wealth[0]),
                               np.mean(index_agent_wealth),
                               index_agent_wealth[-1, -1], 100 * index_metrics['ARR'], index_metrics['ASR'], index_metrics['AVOL'],
                               100 * index_metrics['MDD'], index_metrics['CR'], index_metrics['DDR']
                           ))
            values = [
                np.mean(test_record['max_wealth']),
                np.mean(test_record['min_wealth']),
                np.mean(test_record['avg_wealth']),
                np.mean(test_record['final_wealth']),
                np.mean(test_record['ARR']),
                np.mean(test_record['ASR']),
                np.mean(test_record['AVOL']),
                np.mean(test_record['MDD']),
                np.mean(test_record['CR']),
                np.mean(test_record['DDR'])
            ]


            logger.warning(', '.join(map(str, values)))

            #for test

            agent_wealth, rho_record, weight_record, future_p = agent.evaluation('test')
            metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
            logger.warning('last model metrics:')
            logger.warning(' %.4f, %.4f, %.4f, %.4f,%.3f,  %.3f, %.3f, %.2f, %.3f, %.3f'
                           % (max(agent_wealth[0]), min(agent_wealth[0]),
                               np.mean(agent_wealth),
                               agent_wealth[-1, -1], 100 * metrics['ARR'], metrics['ASR'], metrics['AVOL'],
                               100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                           ))

            torch.save(actor, os.path.join(model_save_dir, 'last_model-' + str(epoch) + '.pkl'))
            np.save(img_dir + f'/last_agent_wealth_{func_args.exp_num}.npy', agent_wealth)
            np.save(img_dir+f'/last_rho_record.npy_{func_args.exp_num}',np.array(rho_record))
            wandb_obj.log({'test/apr': metrics['APR'], 'epoch_step': epoch})

            wandb_obj.log({'test/mdd': metrics['MDD'], 'epoch_step': epoch})

            wandb_obj.log({'test/avol': metrics['AVOL'], 'epoch_step': epoch})

            wandb_obj.log({'test/asr': metrics['ASR'], 'epoch_step': epoch})

            wandb_obj.log({'test/sor': metrics['DDR'], 'epoch_step': epoch})

            wandb_obj.log({'test/cr': metrics['CR'], 'epoch_step': epoch})

            wandb_obj.log({'test/min_wealth': min(agent_wealth[0]), 'epoch_step': epoch})

            wandb_obj.log({'test/max_wealth': max(agent_wealth[0]), 'epoch_step': epoch})

            wandb_obj.log({'test/avg_wealth': np.mean(agent_wealth), 'epoch_step': epoch})

            wandb_obj.log({'test/final_wealth': agent_wealth[-1, -1], 'epoch_step': epoch})



        except KeyboardInterrupt:
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')
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


