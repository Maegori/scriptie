import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
import random
import os
import math

from util.Trainer import Trainer
from util.Tester import Tester
from util.midi import samples_to_midi

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_NOTES = 96
NOTES_PER_MEASURE = 96
MIDI_MAT = 96 * 96
N_MEASURES = 16

CHUNK = False     # wether or not to rechunk the data, set to True if its the first time training
N_EPOCHS = 2500
BATCH_SIZE = 512
SEED = 42
LR = 1e-3
DO_RATE = 0.1
BN_M = 0.9

ROOT_PATH = "MidiFiles/"
MOD_PATH = "./models/VAE_musicgen_model"

class MultiEpochsDataLoader(DataLoader):
    """ Dataloader and sampler (class below this one) which don't have to initialize at each epoch, credit to:
    https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
    will bottleneck if data loading in Dataset is too slow, see MidiDataset._chunker() for more info.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MidiDataset(Dataset):

    def __init__(self, root_dir, batch_size, chunker=True,transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.folder = os.listdir(self.root_dir)
        self.shuffle = shuffle

        if chunker:
            self._chunker()

        if not os.path.exists("data_temp"):
            print("data_temp folder not found, make sure to chunk the data first with CHUNK=True on first run.")
            sys.exit()
        else:
            self.batch_files = os.listdir("data_temp")


    def _chunker(self):
        """Chunks the data into batch_size chunks to reduce IO, 
        will delete previously chunked batches if called.
        """

        if not os.path.exists("data_temp"):
            os.makedirs("data_temp")
        else:
            print("Deleting previous chunks...")
            
            files = os.listdir("data_temp")
            for f in files:
                os.remove(os.path.join("data_temp", f))
        
        if self.shuffle:
            random.shuffle(self.folder)
            folder = self.folder
        else:
            folder = self.folder

        print("Creating chunks of size:", self.batch_size)

        n_batches = math.floor(len(folder) / self.batch_size)
        rem = len(folder) % self.batch_size

        # chunk individual arrays together into a single batch
        for b in range(n_batches):
            batch = torch.empty((self.batch_size, N_MEASURES, NOTES_PER_MEASURE, N_NOTES))
            for i in range(self.batch_size):
                with open(self.root_dir + folder[i + b * self.batch_size], "rb") as tp:
                    batch[i] = torch.load(tp)
            
            with open(f"data_temp/batch{b}.tp", "wb") as tp:
                torch.save(batch, tp)

        # chunk the remainder together into a single batch
        if rem == 0:
            print("Chunking done")
        else:
            batch = torch.empty((rem, N_MEASURES, NOTES_PER_MEASURE, N_NOTES))
            for r in range(rem):
                with open(self.root_dir + folder[r + n_batches * self.batch_size], "rb") as tp:
                    batch[r] = torch.load(tp)

            with open(f"data_temp/batch{n_batches + 1}.tp", "wb") as tp:
                torch.save(batch, tp)

            print("chunking done")

        self.batch_files = os.listdir("data_temp")
            
    def __len__(self):
        return len(os.listdir("data_temp"))

    def __getitem__(self, idx):
        with open("data_temp/" + self.batch_files[idx], "rb") as tp:
            X = torch.load(tp)
        return X

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class Encoder1(nn.Module):
    """
    Pre-encoder of the network, one is created for each measure.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(MIDI_MAT, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Encoder2(nn.Module):
    """
    Second encoder of the network, all the pre-encoders are brought together into a 
    single latent space here.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3200, 1600), 
            nn.ReLU(),
        )

        self.mu = nn.Linear(1600, 1024)
        self.sigma = nn.Linear(1600, 1024)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

class Decoder1(nn.Module):
    """
    First decoder, we inflate the main latent space.
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(1024, 1600),
            nn.BatchNorm1d(1600, momentum=BN_M),
            nn.ReLU(),
            nn.Dropout(DO_RATE),
            nn.Linear(1600, 200 * N_MEASURES),
            nn.BatchNorm1d(200 * N_MEASURES, momentum=BN_M),
            nn.ReLU(),
            nn.Dropout(DO_RATE)
        )



    def forward(self, x):
        return self.decoder(x)

class Decoder2(nn.Module):
    """
    Second decoder, where we inflate the data back to our MIDI measures.
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(200, 2000),
            nn.modules.BatchNorm1d(16, momentum=BN_M),
            nn.ReLU(),
            nn.Dropout(DO_RATE),
            nn.Linear(2000, MIDI_MAT),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    """
    In this class we bring the four encoders and decoders together with reparameterisation.
    """

    def __init__(self, encoder1, encoder2, decoder1, decoder2):
        super().__init__()

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
    
    def forward(self, x0):
        x1 = self.encoder1(x0).view(x0.shape[0], 3200)
        mu, sigma = self.encoder2(x1)

        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        x2 = eps.mul(std).add_(mu)
        
        x3 = self.decoder1(x2).view(x0.shape[0], N_MEASURES, 200)
        x4 = self.decoder2(x3)
        
        return x4, mu, sigma

    def producer(self, epoch=0, tresh=0.001):
        midi_array = torch.empty((16, MIDI_MAT), device='cpu')
        sample = torch.randn((1, 1024), device='cpu')

        x = self.decoder1(sample).view(1, 16, 200)
        midi_array = self.decoder2(x)

        viable_notes = len(midi_array[midi_array > tresh]) 
        print(viable_notes, "notes above the threshold of:", tresh)
        if viable_notes == 0:
            return 
        samples_to_midi(midi_array.detach().reshape((16, 96, 96)), "output/epoch{0}.mid".format(epoch), tresh)

        return
    
class VAETrainer(Trainer):
    def __init__(self, model, optimizer, criterion, trainloader, testloader, batch_size, logPath, device, sample_func):
        super().__init__(model, optimizer, criterion, trainloader, testloader, batch_size, logPath, device, sample_func)

    def calc_loss(self, x):
        x = x.view(-1, N_MEASURES, MIDI_MAT)
        y, mu, sigma = self.model(x)

        reconLoss = self.crit(y, x)

        KLLoss = 0.5 * torch.sum(torch.exp(sigma) + mu*mu - 1.0 - sigma)

        # print(f"Reconstruction loss: {reconLoss/x.shape(0)}, KL loss: {KLLoss/x.shape(0)}")
        return reconLoss + KLLoss

def collate_wrapper(batch):
    """ Since we handle the batching ourselves, we need to change the collate
    function too.
    """
    return batch[0]

if __name__ == "__main__":

    try:
        sys.argv[1]
    except IndexError:
        print("Usage:\n'python musicgen.py train' to train the model \nor \n'python musicgen.py test' to sample from the model")
        sys.exit()

    model = VAE(Encoder1(), Encoder2(), Decoder1(), Decoder2())
    if sys.argv[1] == 'train':
        data = MidiDataset(ROOT_PATH, BATCH_SIZE, chunker=CHUNK)

        train_len = math.floor(0.8 * len(data))
        test_len = len(data) - train_len

        train_set, test_set = torch.utils.data.random_split(data, [train_len, test_len])

        train_loader = MultiEpochsDataLoader(dataset=train_set, batch_size=1, collate_fn=collate_wrapper, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
        test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, collate_fn=collate_wrapper, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss(reduction='sum')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1, steps_per_epoch=len(train_loader)*BATCH_SIZE, epochs=N_EPOCHS, div_factor=100000, verbose=False)

        trainer = VAETrainer(model, optimizer, criterion, scheduler, train_loader, test_loader, BATCH_SIZE, device=DEVICE, sample_func=model.producer)
        trainer.run(N_EPOCHS, BATCH_SIZE)

    elif sys.argv[1] == 'test':
        tester = Tester(model, MOD_PATH, model.producer)

        tester.sample(tester.epoch)
