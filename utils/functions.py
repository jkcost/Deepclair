import math
import random

import numpy as np
import torch
# import seaborn as sns
# import matplotlib.pyplot as plt
switch2days = {'D': 1, 'W': 5, 'M': 21}

#
# def plot_exp(date, method, index, heatmap_data):
#     palette = sns.color_palette("deep")
#     fg, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 3]})
#
#     # 위쪽 그래프 (일자별 데이터)
#     ax1.plot(date[3:], index[3:], color=palette[0], label='agent_wealth')
#     ax1.plot(date[3:], method[0], color=palette[1], label='method_wealth')
#     ax1.set_title('agent_wealth')
#     ax1.legend()
#
#     # heatmap_data를 월별로 평균 계산
#     df = pd.DataFrame({
#         'date': pd.to_datetime(date[4:]),
#         'heatmap_data': heatmap_data
#     })
#     monthly_data = df.resample('M', on='date').mean()['heatmap_data']
#
#     # 월별로 heatmap 그리기
#     sns.heatmap(monthly_data.values.reshape(-1, 1), cmap="YlGnBu", ax=ax2, cbar=True)
#     ax2.set_xlabel('기간')
#     ax2.set_ylabel('데이터')
#
#     # 월별 xtick 설정
#     ax2.set_xticks(np.arange(len(monthly_data)))
#     ax2.set_xticklabels(monthly_data.index.strftime('%Y-%m'), rotation=45)
#
#     plt.tight_layout()
#     plt.show()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_metrics(agent_wealth,trade_mode, MAR=0.):

    trade_ror = agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1
    gain = agent_wealth[:,-1] - agent_wealth[:,0]
    if agent_wealth.shape[0] == trade_ror.shape[0] == 1:
        agent_wealth = agent_wealth.flatten()
    if trade_mode == 'D':
        factor = 1
        Ny = 251
    elif trade_mode == 'W':
        factor = 5
        Ny = 50
    elif trade_mode == 'M':
        factor = 20
        Ny = 12
    else:
        assert ValueError, 'Please check the trading mode'
    year_periods = trade_ror.shape[-1] * factor / 251
    # AT = np.mean(trade_ror, axis=-1, keepdims=True)
    AT  = ((agent_wealth[0] + gain) / agent_wealth[0])**(1/year_periods) - 1
    VT = np.std(trade_ror, axis=-1, keepdims=True)

    # ARR = AT * Ny
    ARR = AT
    AVOL = VT * math.sqrt(Ny)
    ASR = ARR / AVOL
    drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) /\
                     np.maximum.accumulate(agent_wealth, axis=-1)
    MDD = np.max(drawdown, axis=-1)
    CR = ARR / MDD

    tmp1 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2), axis=-1) / \
           np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp1)
    DDR = ARR / downside_deviation

    metrics = {
        'ARR': ARR,
        'AVOL': AVOL,
        'ASR': ASR,
        'MDD': MDD,
        'CR': CR,
        'DDR': DDR
    }

    return metrics


class market_calculate:
    def __init__(self, df, trade_len,allow_short, fee=0.001):
        self.df = df.iloc[:,1:]
        self.fee = fee
        self.trade_len = trade_len
        self.allow_short = allow_short
        self.num_assets = self.df.shape[-1]
        investment_array = [0] * self.num_assets

        # 각 자산의 투자 비율을 계산하여 배열에 할당
        investment_ratio = 1 / self.num_assets
        for i in range(len(investment_array)):
            investment_array[i] = investment_ratio

        self.w = np.array(investment_array)
        self.v = self.v = np.array([1.])

    def simulate(self):
        agent_wealth = np.array([1.0])

        for i in range(0, len(self.df) - self.trade_len, self.trade_len):
            # trading_day 만큼 데이터 슬라이싱
            data_slice = self.df.iloc[i:i + self.trade_len+1]

            # long과 short 비율을 0.5씩으로 설정
            p = np.array([0.5])

            # 일일 수익률 계산 (다음날 종가 / 오늘 종가 - 1)
            port_return = data_slice.pct_change().values[1:] + 1
            ror = np.prod(port_return, axis=0)
            w0 = np.resize(self.w, (2 * len(self.w)))
            # 포트폴리오 업데이트
            v1 = self._step(w0, ror, p)
            agent_wealth = np.append(agent_wealth, v1)

        return agent_wealth

    def _step(self, w0, ror, p):
        """

        :param w0: (batch, 2 * num_assets)
        :param ror:
        :param p:
        :return:
        """
        # if not self.allow_short:
        #     assert (w0[:, self.num_assets:] == 0).all() and (p==1).all()
        if (p < 0.0).all() and (p > 1.0).all():
            print(f' p error : {p}')
        assert (p >= 0.0).all() and (p <= 1.0).all()
        dw0 = self.w
        dv0 = self.v

        if self.allow_short:
            # === short ===
            dv0_short = dv0 * (1 - p)
            dv0_short_after_sale = dv0_short * (1 - self.fee)
            dv0_long = (dv0 * p + dv0_short_after_sale)

            dw0_long = dw0 * ((dv0 * p) / dv0_long)[..., None]
            dw0_long_sale = np.clip((dw0_long - w0[:self.num_assets]), 0., 1.)

            mu0_long = dw0_long_sale.sum(axis=-1) * self.fee
            dw1 = (ror * w0[ :self.num_assets]) / np.sum(ror * w0[ :self.num_assets], axis=-1, keepdims=True)

            LongPosition_value = dv0_long * (1 - mu0_long) * (np.sum(ror * w0[ :self.num_assets], axis=-1))
            ShortPosition_value = dv0_short * (np.sum(ror * w0[ self.num_assets:], axis=-1))

            LongPosition_gain = LongPosition_value - dv0_long
            ShortPosition_gain = dv0_short - ShortPosition_value

            LongPosition_return = LongPosition_gain / (dv0_long + 1e-20)
            ShortPosition_return = ShortPosition_gain / (dv0_short + 1e-20)

            dv1 = LongPosition_value - (ShortPosition_value) / (1 - self.fee) \
                  + dv0_short

            rate_of_return = dv1 / dv0 - 1
            cash_value = 0.
            stocks_value = dv1

        # === only long ===
        else:
            dv0 = self.v
            dw0 = self.w

            mu0_long = self.fee * (np.sum(np.abs(dw0 - w0[:self.num_assets]), axis=-1))

            dw1 = (ror * w0[:, :self.num_assets]) / np.sum(ror * w0[:self.num_assets], axis=-1, keepdims=True)

            LongPosition_value = dv0 * (1 - mu0_long) * np.sum(ror * w0[:self.num_assets], axis=-1)

            LongPosition_gain = LongPosition_value - dv0

            LongPosition_return = LongPosition_gain / (dv0 + 1e-20)

            dv1 = LongPosition_value

            rate_of_return = dv1 / dv0 - 1

            cash_value = 0.
            stocks_value = LongPosition_value

        return dv1

