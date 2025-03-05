import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from api import STS
import librosa
from utils import load_wav_to_torch_librosa as load_wav_to_torch
from mel_processing import spectrogram_torch, mel_spectrogram_torch
import random
import soundfile
logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
    # bufferLoader
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch_RL, spec_to_mel_torch,mel_spectrogram_torch
from download_utils import load_pretrain_model
import torchaudio
from datamodule import DataModule
from pathlib import Path
from torch.distributions import Normal
from text.symbols import symbols
from numpy import dot
from numpy.linalg import norm
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encontered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
global_step = 0


def run():
    hps = utils.get_hparams()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        # backend="nccl",
        init_method="env://",  # Due to some training problem,we proposed to use gloo instead of nccl.
        rank=local_rank,
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    n_gpus = dist.get_world_size()

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    global global_step

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )


    collate_fn = TextAudioSpeakerCollate()

    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    print('outside of loader')

    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(rank)
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder")


    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=0,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(rank)


    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    hps.pretrain_G = hps.pretrain_G 
    hps.pretrain_D = hps.pretrain_D 
    hps.pretrain_dur = hps.pretrain_dur 
    cwd = Path.cwd() 


    if hps.pretrain_G:
        print('load pretrain G')
        print('hps.model_dir',hps.model_dir)
        print('hps.pretrain_G',hps.pretrain_G)
        utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                None,
                skip_optimizer=True
            )

    if hps.pretrain_D:
        print('load pretrain D')
        utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                None,
                skip_optimizer=True
            )
        
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
        if hps.pretrain_dur:
            utils.load_checkpoint(
                    utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                    net_dur_disc,
                    None,
                    skip_optimizer=True
                )
                
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        
        global_step_pre=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step_pre=f'{global_step_pre}'
        global_step_pre=global_step_pre.split('_')[-1]
        global_step_pre=global_step_pre.split('.')[0]
        if hps.pretrain_G:
            global_step = int(global_step_pre)

        else:
            global_step = (epoch_str - 1) * len(train_loader)
        epoch_str = max(epoch_str, 1)
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
    scaler = torch.amp.GradScaler(enabled=hps.train.fp16_run)



    for epoch in tqdm(range(epoch_str, hps.train.epochs + 1), desc="Epoch Progress"):

        try:
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g,net_d,net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler, 
                    [train_loader, eval_loader],
                    logger,
                    [writer, writer_eval],
                )

            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g,net_d,net_dur_disc],
                    [optim_g, optim_d, optim_dur_disc],
                    [scheduler_g, scheduler_d, scheduler_dur_disc],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                )

        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


# Function to pad tensors to the maximum length in the last dimension
def pad_to_max_length(tensor, max_len):
    pad_size = max_len - tensor.shape[-1]
    pad_dims = [0, pad_size]  # Padding for the last dimension
    return F.pad(tensor, pad_dims, mode='constant', value=0)
        
        


def pad_tensors(tensors, pad_value=0):
    # Find the maximum length in the last dimension
    max_len = max(tensor.size(-1) for tensor in tensors)
    
    # Create a list to store the padded tensors
    padded_tensors = []
    
    for tensor in tensors:
        # Calculate the amount of padding needed
        pad_len = max_len - tensor.size(-1)
        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors
    stacked_tensors = torch.stack(padded_tensors)
    return stacked_tensors


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    # print('enter train_and_eval111')
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc= schedulers
    train_loader,eval_loader = loaders


    if writers is not None:

        writer, writer_eval = writers

    
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        goal_spec = torch.tensor(data[0]).cuda(rank, non_blocking=True)
        goal_spec_len = torch.tensor(data[1]).cuda(rank, non_blocking=True)
        goal_wav = torch.tensor(data[2]).cuda(rank, non_blocking=True)
        goal_wav_len = torch.tensor(data[3]).cuda(rank, non_blocking=True)
        goal_emo_id = torch.tensor(data[4]).cuda(rank, non_blocking=True)
        audiopath=data[5]
        input_spec = torch.tensor(data[6]).cuda(rank, non_blocking=True)
        input_spec_len = torch.tensor(data[7]).cuda(rank, non_blocking=True)
        input_wav = torch.tensor(data[8]).cuda(rank, non_blocking=True)
        input_wav_len = torch.tensor(data[9]).cuda(rank, non_blocking=True)
        input_emo_id = torch.tensor(data[10]).cuda(rank, non_blocking=True)
        inputaudio=data[11]
        sid=torch.tensor(data[12]).cuda(rank, non_blocking=True)
        emo_goal_emb = torch.tensor(data[13]).cuda(rank, non_blocking=True)
        emo_input_emb = torch.tensor(data[14]).cuda(rank, non_blocking=True)

        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        with autocast(enabled=hps.train.fp16_run):

            
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p,m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (z,z_p, m_q, logs_q),mu,logs
            ) = net_g(
                input_spec,
                input_spec_len,
                goal_spec,
                goal_spec_len,
                emo_goal_emb,
                emo_input_emb,
            )
            print('y_hat',y_hat.shape)

            mel = spec_to_mel_torch(
                goal_spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            
            y = commons.slice_segments(
                goal_wav, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # emo_y_d_hat_r, emo_y_d_hat_g, _, _ = net_d(y_goal, y_gen.detach()) ##Emotion loss
            
            with torch.amp.autocast(device_type='cuda',enabled=False):

                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )

                loss_disc_all = loss_disc
                

            
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw.detach(), logw_.detach()
                )
  
                with torch.amp.autocast(device_type='cuda',enabled=False):
                    
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc

                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with torch.amp.autocast(device_type='cuda', enabled=hps.train.fp16_run):         
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with torch.amp.autocast(device_type='cuda',enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm= feature_loss(fmap_r, fmap_g)
                
                loss_gen,losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl 

                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
  
                
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                print('reward_for_tensorboard2381')


                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )

                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
                print('line1268')

            if global_step % hps.train.eval_interval == 0:
                evaluate(rank,hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )

                keep_ckpts = getattr(hps.train, "keep_ckpts", 5)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    print('train_and_eval finish')
   
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))
    torch.cuda.empty_cache()



def log_tensor_values(tensor, tensor_name,logger,global_step, lr):
    for i, value in enumerate(tensor):
        print('log_tensor_values value',value)
        print('log_tensor_values global_step',global_step)
        print('log_tensor_values lr',lr)

        logger.info(f'{tensor_name}[{i}] = {value} + at global step {global_step}, leanring rate is {lr}')           



def evaluate(rank,hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
    with torch.no_grad():

        for batch_idx, data in enumerate(tqdm(eval_loader)):
            goal_spec = torch.tensor(data[0]).cuda(rank, non_blocking=True)
            goal_spec_len = torch.tensor(data[1]).cuda(rank, non_blocking=True)
            goal_wav = torch.tensor(data[2]).cuda(rank, non_blocking=True)
            goal_wav_len = torch.tensor(data[3]).cuda(rank, non_blocking=True)
            goal_emo_id = torch.tensor(data[4]).cuda(rank, non_blocking=True)
            audiopath=data[5]
            input_spec = torch.tensor(data[6]).cuda(rank, non_blocking=True)
            input_spec_len = torch.tensor(data[7]).cuda(rank, non_blocking=True)
            input_wav = torch.tensor(data[8]).cuda(rank, non_blocking=True)
            input_wav_len = torch.tensor(data[9]).cuda(rank, non_blocking=True)
            input_emo_id = torch.tensor(data[10]).cuda(rank, non_blocking=True)
            inputaudio=data[11]
            sid=torch.tensor(data[12]).cuda(rank, non_blocking=True)
            emo_goal_emb = torch.tensor(data[13]).cuda(rank, non_blocking=True)
            emo_input_emb = torch.tensor(data[14]).cuda(rank, non_blocking=True)
            
            for use_sdp in [True, False]:
                y_hat, attn,mask, *_ = generator.module.infer(
                    input_spec,
                    input_spec_len,
                    emo_goal_emb,
                    emo_input_emb,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                    y=goal_spec,
                )
 

                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    input_spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )

                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )

                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )

                audio_dict.update({f"gt/audio_{batch_idx}": input_wav[0, :, : input_wav_len[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )

    generator.train()
    print('Evauate done')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
