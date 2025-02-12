import math
import torch
from torch import nn
from torch.nn import functional as F

# from MeloTTS.melo import commons
# from MeloTTS.melo import modules
# from MeloTTS.melo import attentions
import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from commons import init_weights, get_padding
import monotonic_align as monotonic_align
import librosa
from mel_processing import spectrogram_torch
from itertools import zip_longest



class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur,g)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):

        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)

        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
            self.emo_emb_linear = nn.Linear(self.gin_channels, self.filter_channels)
    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        # print('st_dur215_x')

        x = torch.detach(x)
        # print('st_dur216_x')

        x = self.pre(x)
        # print('st_dur218_x')

        if g is not None:
            #print('st_dur219_emo',emo)

            g=torch.detach(g)
            #print('st_dur220_emo',emo)

            x = x +self.cond(g)
            #print('st_dur221_X',x)

        x = self.convs(x, x_mask)
        #print('st_dur222_x',x)

        x = self.proj(x) * x_mask
        #print('st_dur223_x',x)

        if not reverse:
            # #print('st_dur225')

            flows = self.flows
            # #print('st_dur226')

            assert w is not None
            # #print('st_dur227')

            logdet_tot_q = 0
            ##print('st_dur228')

            h_w = self.post_pre(w)
            ##print('st_dur230')

            h_w = self.post_convs(h_w, x_mask)
            ##print('st_dur232')

            h_w = self.post_proj(h_w) * x_mask
            ##print('st_dur234')

            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )


            ##print('st_dur241')

            z_q = e_q
            ##print('st_dur243')

            
            for flow in self.post_flows:
                ##print('st_dur246')

                g=(x + h_w)
                #print('st_dur248')
                #print('z_q post_flows not reserve',z_q)
                #print('emo post_flows not reserce', emo)
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                #print('st_dur250')

                #0103
                #2025
                size_dim0 = x.size(0)
                #print('st_dur251')

                if logdet_q.size(0) != size_dim0:
                    logdet_q = logdet_q.expand(size_dim0).clone()



                
                #print('st_dur257')

                logdet_tot_q += logdet_q


            #print('st_dur261')

            z_u, z1 = torch.split(z_q, [1, 1], 1)
            #print('st_dur264')


            u = torch.sigmoid(z_u) * x_mask
            #print('st_dur268')



            z0 = (w - u) * x_mask
            #print('st_dur273')

            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            #print('st_dur278')


            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )
            #print('st_dur285')


            logdet_tot = 0
            #print('st_dur289')


            z0, logdet = self.log_flow(z0, x_mask)
            #print('st_dur293')

            logdet_tot += logdet
            #print('st_dur296')



            z = torch.cat([z0, z1], 1)
            #print('st_dur301')

            #print('st_dur_input',z)

            for flow in flows:
                ##print('st_dur306')
                #print('zinstochasticdurationpredictor not reverse',z)
                #print('emoinstochasticdurationpredictor not reverse',x)

                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                #print('st_dur310')

                logdet_tot = logdet_tot + logdet
                #print('st_dur313')

            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            #print('st_dur319')




            return nll + logq  # [b]
        
        else:
            #print('st_dur327')

            flows = list(reversed(self.flows))
            #print('st_dur330')


            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            #print('st_dur334')



            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            #print('st_dur342')



            for flow in flows:
                #print('st_dur347')

                #print('zinstochasticdurationpredictor reverse',z)
                #print('emoinstochasticdurationpredictor reverse',x)

                z = flow(z, x_mask, g=x, reverse=reverse)

            #print('st_dur352')

            z0, z1 = torch.split(z, [1, 1], 1)
            #print('st_dur355')

            logw = z0
            #print('st_dur358')

            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask,g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
        num_languages=None,
        num_tones=None,
    ):
        super().__init__()
        if num_languages is None:
            from text import num_languages
        if num_tones is None:
            from text import num_tones
        self.n_vocab = n_vocab
        # self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)

        # nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        self.ja_bert_proj = nn.Conv1d(768, hidden_channels, 1)
        # ###add emotion enbedding
        self.emo_proj = nn.Linear(1024, hidden_channels)

        # self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # self.emo_proj = nn.Linear(1024, hidden_channels)


    def forward(self, x, x_lengths,emo,tone, language, bert, ja_bert):
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        print('505')
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        print('507')
        print("x dtype:", x.dtype)
        print("tone dtype:", tone.dtype)
        print("language dtype:", language.dtype)
        print("bert_emb dtype:", bert_emb.dtype)
        print("ja_bert_emb dtype:", ja_bert_emb.dtype)
        print("emo dtype:", emo.dtype)
        print("self.tone_emb(tone) dtype:", self.tone_emb(tone).dtype)
        print("self.language_emb(language) dtype:", self.language_emb(language).dtype)


        emo = emo.float()
        x = (self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            +self.emo_proj(emo.unsqueeze(1))) * math.sqrt(self.hidden_channels) # [b, t, h]
        print('515')
        x = torch.transpose(x, 1, -1) # [b, h, t]
        print('517')
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        print('520')
        x = self.encoder(x * x_mask, x_mask)
        print('521')
        
        stats = self.proj(x) * x_mask
        print('522')
        
        m, logs = torch.split(stats, self.out_channels, dim=1)
        print('527')
        
        
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None,reverse=False):

        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g,reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        ###add emotion enbedding
        # self.emo_proj = nn.Linear(1024, gin_channels)

    def forward(self, x, x_lengths,g, tau=1.0):
        # #print('posterencoder613')
        
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        
        # print('posterencoder618')

        x = self.pre(x) * x_mask

        # print('x',x.shape)
        # print('emo',emo.shape)
        # x=x+emo
        # print('x',x.shape)

        x = self.enc(x, x_mask, g=g)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)


        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        # print('posterencoder635')

        return z, m, logs, x_mask
    
class InputAudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        ###add emotion enbedding
        # self.emo_proj = nn.Linear(1024, gin_channels)

    def forward(self, x, x_lengths, tau=1.0):
        # #print('posterencoder613')
        
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        
        # print('posterencoder618')

        x = self.pre(x) * x_mask

        # print('x',x.shape)
        # print('emo',emo.shape)
        # x=x+emo
        # print('x',x.shape)

        x = self.enc(x, x_mask,)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)


        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        # print('posterencoder635')

        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 2, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g,):

        x = self.conv_pre(x)

        # if g is not None:
        #     x = x + self.cond(g)
        if g is not None:
            x = x + self.cond(g)


        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        # x = self.conv_post(x)
        # x = torch.tanh(x)

        # return x
        stats =self.conv_post(x)
        mu,logs=torch.split(stats,1,dim=1)

        x2=mu+torch.randn_like(mu)*torch.exp(logs)
        
        return x2, mu, logs

    def compute_log_probability(self, x, mu, logs):
        # Compute the standard deviation from logs
        sigma = torch.exp(logs)
    
        # Compute the variance
        var = sigma ** 2
    
        # Compute the log probability
        log_prob = -0.5 * ((x - mu) ** 2 / var + torch.log(2 * torch.pi * var))
    
        # Sum the log probabilities over the last dimension (feature dimension)
        log_prob = log_prob.sum(dim=-1)
    
        return log_prob
    
    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=False):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
            print('[Ref Enc]: using layer norm')
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=6,
        flow_share_parameter=False,
        use_transformer_flow=True,
        use_vc=False,
        num_languages=None,
        num_tones=None,
        norm_refenc=False,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
            num_languages=num_languages,
            num_tones=num_tones,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.enc_input = InputAudioEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )


        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        self.n_speakers = n_speakers
        if n_speakers == 0:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)
        else:
            self.enc_p = TextEncoder(n_vocab,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout)
            self.sdp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_vc = use_vc
        self.emo_proj = nn.Linear(1024, gin_channels)
        # self.emo_proj = nn.Linear(1024, hidden_channels)



    def forward(self,
                y, y_lengths,
                goal_spec,
                goal_spec_len,
                emo,
                emo_input,
):
        print('enter syn')
        print('emo',emo.type())
        emo=emo.float()
        
        emo=self.emo_proj(emo.unsqueeze(0))
        emo=torch.transpose(emo,0,1)
        emo=torch.transpose(emo,1,2)
        emo_input=emo_input.float()
        
        emo_input=self.emo_proj(emo_input.unsqueeze(0))
        emo_input=torch.transpose(emo_input,0,1)
        emo_input=torch.transpose(emo_input,1,2)
        # if self.n_speakers > 0:
        #     print('run emb_g')
        #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        # else:
        g = self.ref_enc(goal_spec.transpose(1, 2)).unsqueeze(-1)
            
        print('y',y.type())
        print('y_lengths',y_lengths.type())
        y=y.float()
        y_lengths=y_lengths.float()
        x, m_p, logs_p, x_mask = self.enc_input(y, y_lengths, )


        z, m_q, logs_q, y_mask = self.enc_q(goal_spec, goal_spec_len,g=g, )

        print('syn1041')

        z_p = self.flow(z, y_mask,g=g)


        print('syn1042')
        # input()
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            # #print('syn1070')

            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            # print('syn1074')

            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # print('syn1078')

            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # print('syn1082')

            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            # print('syn1086')

            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            # print('syn1088')

            if self.use_noise_scaled_mas:
                # print('syn1090')

                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )

                neg_cent = neg_cent + epsilon
            # print('syn1076')

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # print('syn1079')

            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        #     print('syn1086')
        # print('syn1087')

        w = attn.sum(2)
        # print('syn1090')

        ###remove sdp
        # #print('g',g)
        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        # print('syn1093')

        l_length_sdp = l_length_sdp / torch.sum(x_mask)
        # print('syn1096')

        logw_ = torch.log(w + 1e-6) * x_mask
        # print('syn1099')

        logw = self.dp(x, x_mask,g)
        # print('syn1102')

        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging
        # print('syn1107')

        l_length = l_length_dp + l_length_sdp
        # print('syn1110')

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        # print('syn1114')

        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        # print('syn1117')

        z_slice, ids_slice = commons.rand_slice_segments(
            z, goal_spec_len, self.segment_size
        )
        # print('syn1108')

        o, mu, logs = self.dec(z_slice, g)
  
        return (
            o,
            # o_unsliced,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
            (z, z_p, m_q, logs_q), 
            mu,logs
        )

    def forward_change(self,
            y, y_lengths,
            goal_spec,
            goal_spec_len,
            emo,
            emo_input,
):
        print('enter syn')
        print('emo', emo.type())
        emo = emo.float()
        
        emo = self.emo_proj(emo.unsqueeze(0))
        emo = torch.transpose(emo, 0, 1)
        emo = torch.transpose(emo, 1, 2)
        emo_input = emo_input.float()
        
        emo_input = self.emo_proj(emo_input.unsqueeze(0))
        emo_input = torch.transpose(emo_input, 0, 1)
        emo_input = torch.transpose(emo_input, 1, 2)

        g = self.ref_enc(goal_spec.transpose(1, 2)).unsqueeze(-1)
        g_y = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            
        print('y', y.type())
        print('y_lengths', y_lengths.type())

        # Encode both source (`y`) and reference (`goal_spec`)
        x, m_p, logs_p, x_mask = self.enc_q(y, y_lengths, g=g_y)
        z, m_q, logs_q, y_mask = self.enc_q(goal_spec, goal_spec_len, g=g)

        print('syn1041')

        # Flow transformation to remove emotion features
        z_p = self.flow(z, y_mask, g=g)
        z_p_y = self.flow(x, x_mask, g=g_y)

        print('syn1042')

        # Compute monotonic alignment
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            # #print('syn1070')

            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            # print('syn1074')

            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # print('syn1078')

            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # print('syn1082')

            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            # print('syn1086')

            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            # print('syn1088')

            if self.use_noise_scaled_mas:
                # print('syn1090')

                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )

                neg_cent = neg_cent + epsilon
            # print('syn1076')

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # print('syn1079')

            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        #     print('syn1086')
        # print('syn1087')

        w = attn.sum(2)
        # print('syn1090')

        ###remove sdp
        # #print('g',g)
        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        # print('syn1093')

        l_length_sdp = l_length_sdp / torch.sum(x_mask)
        # print('syn1096')

        logw_ = torch.log(w + 1e-6) * x_mask
        # print('syn1099')

        logw = self.dp(x, x_mask,g)
        # print('syn1102')

        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging
        # print('syn1107')

        l_length = l_length_dp + l_length_sdp
        # print('syn1110')

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        # print('syn1114')

        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        # print('syn1117')

        z_slice, ids_slice = commons.rand_slice_segments(
            z, goal_spec_len, self.segment_size
        )
        # print('syn1108')

        o, mu, logs = self.dec(z_slice, g)
  
        return (
            o,
            # o_unsliced,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
            (z, z_p, m_q, logs_q), 
            mu,logs
        )



    def infer_sts(
        self,
        # x,
        # x_lengths,
        audio_src_path,
        # sid,
        emo,
        sampling_rate,
        filter_length,
        hop_length, 
        win_length,
        g=None, 
        noise_scale_w=0.8,
        sdp_ratio=0,
        length_scale=1,
        max_len=None,
        noise_scale=0.667,
        y=None):
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        # print('audio_src_path',audio_src_path)
        audio, sample_rate = librosa.load(audio_src_path, sr=sampling_rate)
        audio = torch.tensor(audio).float()
        with torch.no_grad():
            y = torch.FloatTensor(audio)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, filter_length,
                                    sampling_rate, hop_length, win_length,
                                    center=False)
            spec_lengths = torch.LongTensor([spec.size(-1)])
        # if g is None:
        #     if self.n_speakers > 0:
        #         g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        #     else:
        #         g = self.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
        # if self.use_vc:
        #     g_p = None
        # else:
        #     g_p = g
        emo=self.emo_proj(emo.unsqueeze(0))
        emo=emo.unsqueeze(-1)
        # emo.squeeze(0).squeeze(1).unsqueeze(-1)


        spec=spec.to(emo.device)
        spec_lengths = spec_lengths.to(emo.device)
        x, m_p, logs_p, x_mask = self.enc_q(spec, spec_lengths,emo,)
        logw = self.sdp(x, x_mask, emo=emo, reverse=True, noise_scale=noise_scale_w) * (
                sdp_ratio
            ) + self.dp(x, x_mask,emo) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                x_mask.dtype
            )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
                1, 2
            )
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                1, 2
            )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask,  emo=emo, reverse=True)
        o, mu, logs = self.dec((z * y_mask)[:, :, :max_len],emo=emo)

        # z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, emo,g=g, tau=1)

        # z_p = self.flow(z, y_mask, g=g,emo=emo)
        # z_hat = self.flow(z_p, y_mask, g=g, emo=emo, reverse=True)
        # o_hat = self.dec(z_hat * y_mask, emo, g=g)
        return o,  y_mask, (z, z_p, m_p, logs_p)

    def infer_sts_new(
            self,
            audio_src_path,
            emo,
            emo_tgt,
            emo_path,
            sampling_rate,
            filter_length,
            hop_length, 
            win_length,
            g=None, 
            noise_scale_w=0.8,
            sdp_ratio=0,
            length_scale=1,
            max_len=None,
            noise_scale=0.667,
            y=None):
            # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
            # g = self.gst(y)
            # print('audio_src_path',audio_src_path)
            # if self.n_emos > 0:
            #     emo = self.emb_g(emo).unsqueeze(-1)  # [b, h, 1]
            # else:
            #     emo = emo
            
            
            emo=emo.float()
            emo=self.emo_proj(emo.unsqueeze(0))

            emo=emo.unsqueeze(-1)


            # emo=emo.squeeze(0)
            emo_tgt=emo_tgt.float()
            emo_tgt=self.emo_proj(emo_tgt.unsqueeze(0))

            emo_tgt=emo_tgt.unsqueeze(-1)


            # emo_tgt=emo_tgt.squeeze(0)
            # emo=None
            # g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            
            audio, sample_rate = librosa.load(audio_src_path, sr=sampling_rate)
            audio_emo, sample_rate_emo = librosa.load(emo_path, sr=sampling_rate)
            
            audio = torch.tensor(audio).float()
            audio_emo = torch.tensor(audio_emo).float()
            
            with torch.no_grad():
                y = torch.FloatTensor(audio)
                y = y.unsqueeze(0)
                y=y.to(emo.device)
                spec = spectrogram_torch(y, filter_length,
                                        sampling_rate, hop_length, win_length,
                                        center=False)
                spec_lengths = torch.LongTensor([spec.size(-1)])
                spec=spec.to(emo.device)
                spec_lengths = spec_lengths.to(emo.device)
                
                y_emo = torch.FloatTensor(audio_emo)
                y_emo = y_emo.unsqueeze(0)
                y_emo=y_emo.to(emo.device)
                spec_emo = spectrogram_torch(y_emo, filter_length,
                                        sampling_rate, hop_length, win_length,
                                        center=False)
                spec_lengths_emo = torch.LongTensor([spec_emo.size(-1)])
                spec_emo=spec_emo.to(emo.device)
                spec_lengths_emo = spec_lengths_emo.to(emo.device)
                g_emo = self.ref_enc(spec_emo.transpose(1, 2)).unsqueeze(-1)
                g_input = self.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
                
                
                
                x, m_p, logs_p, x_mask = self.enc_q(spec, spec_lengths,g_input,tau=0.3)

                z = self.flow(x, x_mask,  g=g_input, reverse=False)
                z_hat = self.flow(z, x_mask,  g=g_emo, reverse=True)
                o, mu, logs = self.dec((z_hat * x_mask),g=g_emo)
                # o, mu, logs = self.dec((z_hat * x_mask)[:, :, :max_len],emo=emo_tgt)
               
                
            return o,  x_mask, (x, z, z_hat)
    def infer_sts_train(
            self,
            # x,
            # x_lengths,
            input_spec,
            input_spec_len,
            # sid,
            emo,
            emo_tgt,
            sampling_rate,
            filter_length,
            hop_length, 
            win_length,
            g=None, 
            noise_scale_w=0.8,
            sdp_ratio=0,
            length_scale=1,
            max_len=None,
            noise_scale=0.667,
            y=None):

            spec=input_spec.to(emo.device)
            spec_lengths = input_spec_len.to(emo.device)
            x, m_p, logs_p, x_mask = self.enc_q(spec, spec_lengths,emo,)

            z = self.flow(x, x_mask,  emo=emo, reverse=False)
            z_hat = self.flow(z, x_mask,  emo=emo_tgt, reverse=True)
            
            o, mu, logs = self.dec((z_hat * x_mask)[:, :, :max_len],emo=emo_tgt)
            
            
            return o,  x_mask, (x, z, z_hat)

    def infer(
            self,
            x,
            x_lengths,
            emo,
            emo_input,
            noise_scale_w=0.8,
            sdp_ratio=0,
            length_scale=1,
            max_len=None,
            noise_scale=0.667,
            y=None
             ):

            emo=emo.float()
            emo=self.emo_proj(emo.unsqueeze(0))

            emo=emo.unsqueeze(-1)


            emo=emo.squeeze(0)
            
            emo_input=emo_input.float()
            emo_input=self.emo_proj(emo_input.unsqueeze(0))

            emo_input=emo_input.unsqueeze(-1)


            emo_input=emo_input.squeeze(0)
            x=x.to(emo.device)
            x_lengths = x_lengths.to(emo.device)
            # emo=None
            # if self.n_speakers > 0:
            #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            # else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                
            x, m_p, logs_p, x_mask = self.enc_input(x, x_lengths, )

            logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
                sdp_ratio
            ) + self.dp(x, x_mask,emo) * (1 - sdp_ratio)

            w = torch.exp(logw) * x_mask * length_scale

            w_ceil = torch.ceil(w)


            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()


            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                x_mask.dtype
            )


            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)


            attn = commons.generate_path(w_ceil, attn_mask)


            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']


            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']


            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale


            z = self.flow(z_p, y_mask, g=g, reverse=True)


            o, mu, logs = self.dec((z * y_mask)[:, :, :max_len], g=g)



            return o, attn, y_mask, (z, z_p, m_p, logs_p)


            # z, m_q, logs_q, y_mask = self.enc_q(x, x_lengths, emo,g=g, tau=1)

            # z_p = self.flow(z, y_mask, g=g,emo=emo)
            # z_hat = self.flow(z_p, y_mask, g=g, emo=emo, reverse=True)
            # o_hat = self.dec(z_hat * y_mask, emo, g=g)
            # return o_hat,  y_mask, (z, z_p, z_hat)

    def infer_new(
                self,
                audio_src_path,
                emo,
                emo_path,
                sampling_rate,
                filter_length,
                hop_length, 
                win_length,
                noise_scale_w=0.8,
                sdp_ratio=0,
                length_scale=1,
                max_len=None,
                noise_scale=0.667,
                y=None,
                ):
                emo=self.emo_proj(emo.unsqueeze(0))
                emo=emo.unsqueeze(-1)
                
                audio, sample_rate = librosa.load(audio_src_path, sr=sampling_rate)
                audio = torch.tensor(audio).float()
                audio_emo, sample_rate_emo = librosa.load(emo_path, sr=sampling_rate)
                audio_emo = torch.tensor(audio_emo).float()
                
                with torch.no_grad():
                    y = torch.FloatTensor(audio)
                    y = y.unsqueeze(0)
                    y=y.to(emo.device)
                    spec = spectrogram_torch(y, filter_length,
                                            sampling_rate, hop_length, win_length,
                                            center=False)
                    spec_lengths = torch.LongTensor([spec.size(-1)])
                    spec=spec.to(emo.device)
                    spec_lengths = spec_lengths.to(emo.device)
                    y_emo = torch.FloatTensor(audio_emo)
                    y_emo = y_emo.unsqueeze(0)
                    y_emo=y_emo.to(emo.device)
                    spec_emo = spectrogram_torch(y_emo, filter_length,
                                            sampling_rate, hop_length, win_length,
                                            center=False)
                    spec_lengths_emo = torch.LongTensor([spec_emo.size(-1)])
                    spec_emo=spec_emo.to(emo.device)
                    spec_lengths_emo = spec_lengths_emo.to(emo.device)
                    g_emo = self.ref_enc(spec_emo.transpose(1, 2)).unsqueeze(-1)
                    g_input = self.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
                    
                x, m_p, logs_p, x_mask = self.enc_input(spec, spec_lengths,)


                logw = self.sdp(x, x_mask, g=g_emo, reverse=True, noise_scale=noise_scale_w) * (
                    sdp_ratio
                ) + self.dp(x, x_mask,g_emo) * (1 - sdp_ratio)

                w = torch.exp(logw) * x_mask * length_scale

                w_ceil = torch.ceil(w)


                y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()


                y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
                    x_mask.dtype
                )


                attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)


                attn = commons.generate_path(w_ceil, attn_mask)


                m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
                    1, 2
                )  # [b, t', t], [b, t, d] -> [b, d, t']


                logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
                    1, 2
                )  # [b, t', t], [b, t, d] -> [b, d, t']


                z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale


                z = self.flow(z_p, y_mask, g=g_emo, reverse=True)


                o, mu, logs = self.dec((z * y_mask)[:, :, :max_len], g=g_emo)



                return o, attn, y_mask, (z, z_p, m_p, logs_p)


                # z, m_q, logs_q, y_mask = self.enc_q(x, x_lengths, emo,g=g, tau=1)

                # z_p = self.flow(z, y_mask, g=g,emo=emo)
                # z_hat = self.flow(z_p, y_mask, g=g, emo=emo, reverse=True)
                # o_hat = self.dec(z_hat * y_mask, emo, g=g)
                # return o_hat,  y_mask, (z, z_p, z_hat)

    def voice_conversion(self, y, y_lengths,  emo_src,emo_tgt,tau=1):        
        # g_src = sid_src
        # g_tgt = sid_tgt
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths,emo=emo_src,  tau=tau)
        z_p = self.flow(z, y_mask, emo=emo_src,)
        z_hat = self.flow(z_p, y_mask,emo=emo_tgt, reverse=True)
        o_hat, mu, logs = self.dec(z_hat * y_mask, emo_tgt,)
        return o_hat, y_mask, (z, z_p, z_hat)
    
    def extract_se(self, ref_wav_list, hps,language,se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        
        # device = self.device
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(language.device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                        center=False).to(language.device)
            # print('y1819',y.shape)
            
            with torch.no_grad():
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                # g=g.squeeze()
                print('g.shape',g.shape)
                gs.append(g.detach())
        print('before gs')
        device = language.device

        # Move all tensors to the correct device
        gs = [t.to(device) for t in gs]

        # Determine the maximum length
        max_len = max(t.size(0) for t in gs)

        # Pad each tensor to the maximum length with zeros
        padded_tensors = [
            torch.cat([t, torch.zeros(max_len - t.size(0), device=device)]) for t in gs
        ]
        print('before gs1852')
        
        gs = torch.stack(padded_tensors).mean(1)
        print('gs.shape',gs.shape)
        return gs
    
    def extract_se_(self, ref_wav_list, hps,device,se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        
        # device = self.device
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                        center=False).to(device)
            # print('y1819',y.shape)
            
            with torch.no_grad():
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                # g=g.squeeze()
                print('g.shape',g.shape)
                gs.append(g.detach())
        print('before gs')
        device = device

        # Move all tensors to the correct device
        gs = [t.to(device) for t in gs]

        # Determine the maximum length
        max_len = max(t.size(0) for t in gs)

        # Pad each tensor to the maximum length with zeros
        padded_tensors = [
            torch.cat([t, torch.zeros(max_len - t.size(0), device=device)]) for t in gs
        ]
        print('before gs1852')
        
        gs = torch.stack(padded_tensors).mean(1)
        print('gs.shape',gs.shape)
        return gs
    
# class SynthesizerTrn_old(nn.Module):
#     """
#     Synthesizer for Training
#     """

#     def __init__(
#         self,
#         n_vocab,
#         spec_channels,
#         segment_size,
#         inter_channels,
#         hidden_channels,
#         filter_channels,
#         n_heads,
#         n_layers,
#         kernel_size,
#         p_dropout,
#         resblock,
#         resblock_kernel_sizes,
#         resblock_dilation_sizes,
#         upsample_rates,
#         upsample_initial_channel,
#         upsample_kernel_sizes,
#         n_speakers=256,
#         gin_channels=256,
#         use_sdp=True,
#         n_flow_layer=4,
#         n_layers_trans_flow=6,
#         flow_share_parameter=False,
#         use_transformer_flow=True,
#         use_vc=False,
#         num_languages=None,
#         num_tones=None,
#         norm_refenc=False,
#         **kwargs
#     ):
#         super().__init__()
#         self.n_vocab = n_vocab
#         self.spec_channels = spec_channels
#         self.inter_channels = inter_channels
#         self.hidden_channels = hidden_channels
#         self.filter_channels = filter_channels
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.kernel_size = kernel_size
#         self.p_dropout = p_dropout
#         self.resblock = resblock
#         self.resblock_kernel_sizes = resblock_kernel_sizes
#         self.resblock_dilation_sizes = resblock_dilation_sizes
#         self.upsample_rates = upsample_rates
#         self.upsample_initial_channel = upsample_initial_channel
#         self.upsample_kernel_sizes = upsample_kernel_sizes
#         self.segment_size = segment_size
#         self.n_speakers = n_speakers
#         self.gin_channels = gin_channels
#         self.n_layers_trans_flow = n_layers_trans_flow
#         self.use_spk_conditioned_encoder = kwargs.get(
#             "use_spk_conditioned_encoder", True
#         )
#         self.use_sdp = use_sdp
#         self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
#         self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
#         self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
#         self.current_mas_noise_scale = self.mas_noise_scale_initial
#         if self.use_spk_conditioned_encoder and gin_channels > 0:
#             self.enc_gin_channels = gin_channels
#         else:
#             self.enc_gin_channels = 0
#         self.enc_p = TextEncoder(
#             n_vocab,
#             inter_channels,
#             hidden_channels,
#             filter_channels,
#             n_heads,
#             n_layers,
#             kernel_size,
#             p_dropout,
#             gin_channels=self.enc_gin_channels,
#             num_languages=num_languages,
#             num_tones=num_tones,
#         )
#         self.dec = Generator(
#             inter_channels,
#             resblock,
#             resblock_kernel_sizes,
#             resblock_dilation_sizes,
#             upsample_rates,
#             upsample_initial_channel,
#             upsample_kernel_sizes,
#             gin_channels=gin_channels,
#         )
#         self.enc_q = PosteriorEncoder(
#             spec_channels,
#             inter_channels,
#             hidden_channels,
#             5,
#             1,
#             16,
#             gin_channels=gin_channels,
#         )
#         if use_transformer_flow:
#             self.flow = TransformerCouplingBlock(
#                 inter_channels,
#                 hidden_channels,
#                 filter_channels,
#                 n_heads,
#                 n_layers_trans_flow,
#                 5,
#                 p_dropout,
#                 n_flow_layer,
#                 gin_channels=gin_channels,
#                 share_parameter=flow_share_parameter,
#             )
#         else:
#             self.flow = ResidualCouplingBlock(
#                 inter_channels,
#                 hidden_channels,
#                 5,
#                 1,
#                 n_flow_layer,
#                 gin_channels=gin_channels,
#             )
#         self.sdp = StochasticDurationPredictor(
#             hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
#         )
#         self.dp = DurationPredictor(
#             hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
#         )

#         if n_speakers > 0:
#             self.emb_g = nn.Embedding(n_speakers, gin_channels)
#         else:
#             self.ref_enc = ReferenceEncoder(spec_channels, gin_channels, layernorm=norm_refenc)
#         self.use_vc = use_vc


#     def forward(self, y, y_lengths, sid,emo):
#         if self.n_speakers > 0:
#             g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
#         else:
#             g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
#         if self.use_vc:
#             g_p = None
#         else:
#             g_p = g
#         # x, m_p, logs_p, x_mask = self.enc_p(
#         #     x, x_lengths, tone, language, bert, ja_bert,emo, g=g_p
#         # )
#         z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, emo, g=g)
#         z_p = self.flow(z, y_mask, g=g)

#         # with torch.no_grad():
#         #     # negative cross-entropy
#         #     s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
#         #     neg_cent1 = torch.sum(
#         #         -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
#         #     )  # [b, 1, t_s]
#         #     neg_cent2 = torch.matmul(
#         #         -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
#         #     )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
#         #     neg_cent3 = torch.matmul(
#         #         z_p.transpose(1, 2), (m_p * s_p_sq_r)
#         #     )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
#         #     neg_cent4 = torch.sum(
#         #         -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
#         #     )  # [b, 1, t_s]
#         #     neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
#         #     if self.use_noise_scaled_mas:
#         #         epsilon = (
#         #             torch.std(neg_cent)
#         #             * torch.randn_like(neg_cent)
#         #             * self.current_mas_noise_scale
#         #         )
#         #         neg_cent = neg_cent + epsilon

#         #     attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
#         #     attn = (
#         #         monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
#         #         .unsqueeze(1)
#         #         .detach()
#         #     )

#         # w = attn.sum(2)

#         # l_length_sdp = self.sdp(x, x_mask, w, g=g)
#         # l_length_sdp = l_length_sdp / torch.sum(x_mask)

#         # logw_ = torch.log(w + 1e-6) * x_mask
#         # logw = self.dp(x, x_mask, g=g)
#         # l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
#         #     x_mask
#         # )  # for averaging

#         # l_length = l_length_dp + l_length_sdp

#         # # expand prior
#         # m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
#         # logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

#         z_slice, ids_slice = commons.rand_slice_segments(
#             z, y_lengths, self.segment_size
#         )
#         o = self.dec(z_slice, g=g)
#         return (
#             o,
#             l_length,
#             attn,
#             ids_slice,
#             x_mask,
#             y_mask,
#             (z, z_p, m_p, logs_p, m_q, logs_q),
#             (x, logw, logw_),
#         )

#     def infer(
#         self,
#         x,
#         x_lengths,
#         sid,
#         tone,
#         language,
#         bert,
#         ja_bert,
#         emo,
#         noise_scale=0.667,
#         length_scale=1,
#         noise_scale_w=0.8,
#         max_len=None,
#         sdp_ratio=0,
#         y=None,
#         g=None,
#     ):
#         # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
#         # g = self.gst(y)
#         if g is None:
#             if self.n_speakers > 0:
#                 g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
#             else:
#                 g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
#         if self.use_vc:
#             g_p = None
#         else:
#             g_p = g
#         x, m_p, logs_p, x_mask = self.enc_p(
#             x, x_lengths, tone, language, bert, ja_bert, emo,g=g_p
#         )
#         logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
#             sdp_ratio
#         ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
#         w = torch.exp(logw) * x_mask * length_scale
        
#         w_ceil = torch.ceil(w)
#         y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
#         y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
#             x_mask.dtype
#         )
#         attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
#         attn = commons.generate_path(w_ceil, attn_mask)

#         m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
#             1, 2
#         )  # [b, t', t], [b, t, d] -> [b, d, t']
#         logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
#             1, 2
#         )  # [b, t', t], [b, t, d] -> [b, d, t']

#         z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
#         z = self.flow(z_p, y_mask, g=g, reverse=True)
#         o = self.dec((z * y_mask)[:, :, :max_len], g=g)
#         # print('max/min of o:', o.max(), o.min())
#         return o, attn, y_mask, (z, z_p, m_p, logs_p)

#     def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):        
#         g_src = sid_src
#         g_tgt = sid_tgt
#         z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, tau=tau)
#         z_p = self.flow(z, y_mask, g=g_src)
#         z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
#         o_hat = self.dec(z_hat * y_mask, g=g_tgt)
#         return o_hat, y_mask, (z, z_p, z_hat)
