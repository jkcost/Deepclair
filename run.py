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


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)
    if func_args.method =='PG':
        func_args.msu_bool = False

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H%M%S')


    if func_args.mode == 'train':
        PREFIX =  'outpus/'
        PREFIX = os.path.join(PREFIX,start_time)
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

        index_data = pd.read_csv(data_prefix+'^DJI.csv')
        index_data['date'] = pd.to_datetime(index_data['date'])
        # test_idx = index_data[index_data['date'].dt.year >= func_args.test_period].index[0]

        if func_args.market == 'DJIA':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            logger.debug(f'stocks_data.shape:{stocks_data.shape}')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            if func_args.method =='PRED_MSU':
                predict_data = pd.read_csv(data_prefix+'^DJI.csv',index_col=0)
                predict_data = predict_data[['date','Adj Close']]
            else:
                predict_data= pd.read_csv(data_prefix+'DJI_stocks.csv')

            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            # test_idx = 7328 #12107
            # test_idx = 5650
            # test_idx = 4191
            test_idx = int(stocks_data.shape[1] * 0.7)
            allow_short = True
        elif func_args.market == 'HSI':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 4211
            allow_short = True

        elif func_args.market == 'CSI100':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 1944
            market_history = None
            allow_short = False

        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,predict_data=predict_data,
                           in_features=func_args.in_features, val_idx=test_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len,
                           trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)
        ###############modify #######
        A = A[:rate_of_return.shape[0], :rate_of_return.shape[0]]
        ##########################
        supports = [A]
        actor = RLActor(supports, func_args).to(func_args.device)  # define type of network
        agent = RLAgent(env, actor, func_args)

        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        try:
            max_cr = 0
            for epoch in range(func_args.epochs):
                epoch_return = 0
                for j in tqdm(range(mini_batch_num)):
                # for j in tqdm(range(2)):
                    # if j ==7:
                    #     print()
                    episode_return, avg_rho, avg_mdd,loss = agent.train_episode()
                    epoch_return += episode_return
                    print(f'epoch{j} loss -> {loss}')
                avg_train_return = epoch_return / mini_batch_num
                logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f' %
                               (start_time, epoch, avg_train_return, avg_rho, avg_mdd))
                agent_wealth,rho_record,weight_record = agent.evaluation()
                metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
                writer.add_scalar('Test/APR', metrics['APR'], global_step=epoch)
                writer.add_scalar('Test/MDD', metrics['MDD'], global_step=epoch)
                writer.add_scalar('Test/AVOL', metrics['AVOL'], global_step=epoch)
                writer.add_scalar('Test/ASR', metrics['ASR'], global_step=epoch)
                writer.add_scalar('Test/SoR', metrics['DDR'], global_step=epoch)
                writer.add_scalar('Test/CR', metrics['CR'], global_step=epoch)

                if metrics['CR'] > max_cr:
                    print('New Best CR Policy!!!!')
                    invest_record = json.dumps(weight_record)
                    logger.warning(f"stock_lst:{weight_record['stock']}")
                    logger.warning(f"long_ratio:{weight_record['long_ratio']}")
                    logger.warning(f"short_ratio:{weight_record['short_ratio']}")
                    logger.warning(f"trade_len_stock_return:{weight_record['stock_return']}")

                    max_cr = metrics['CR']

                    record_path = os.path.join(save_dir,'invest_record.json')
                    with open(record_path, 'w') as f:
                        json.dump(invest_record, f, indent=4)
                    torch.save(actor, os.path.join(model_save_dir, 'best_cr-' + str(epoch) + '.pkl'))
                logger.warning('after training %d round, max wealth: %.4f, min wealth: %.4f,'
                               ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                               'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                               % (
                                   epoch, max(agent_wealth[0]), min(agent_wealth[0]), np.mean(agent_wealth),
                                   agent_wealth[-1, -1], 100 * metrics['APR'], metrics['ASR'], metrics['AVOL'],
                                   100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                               ))
        except KeyboardInterrupt:
            torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--method', type=str, default='PRED_MSU')
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--test_period', type=int,default=2016)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
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
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=13, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')  # number of input feature
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
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


