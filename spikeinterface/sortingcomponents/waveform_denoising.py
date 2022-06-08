import numpy as np
import numpy.random as random
import os
import scipy
from scipy import signal
import math
from sklearn.decomposition import PCA
try:
    import torch 
    from torch import nn, optim
    import torch.utils.data as Data
    from torch.nn import functional as F
    from torch import distributions
    HAVE_TORCH = True
except:
    HAVE_TORCH = False

if HAVE_TORCH:
    class SingleChanDenoiser(nn.Module):
        def __init__(self, pretrained_path=None, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121):
#             if torch.cuda.is_available():
#                 device = 'cuda:0'
#             else:
#                 device = 'cpu'
#             torch.cuda.set_device(device)
            super(SingleChanDenoiser, self).__init__()
            feat1, feat2 = n_filters
            size1, size2 = filter_sizes
            #TODO For Loop 
            self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU()).to()
            self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
            n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
            self.out = nn.Linear(n_input_feat, spike_size)
            self.pretrained_path = pretrained_path

        def forward(self, x):
            x = x[:, None]
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.shape[0], -1)
            return self.out(x)

        #TODO: Put it outside the class
        def load(self):
            checkpoint = torch.load(self.pretrained_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            return self

        def train(self, fname_save, DenoTD, n_train=50000, n_test=500, EPOCH=2000, BATCH_SIZE=512, LR=0.0001):
            """
            DenoTD instance of Denoise Training Data class
            """
            print('Training NN denoiser')

            if os.path.exists(fname_save):
                return

            optimizer = torch.optim.Adam(self.parameters(), lr=LR)   # optimize all cnn parameters
            loss_func = nn.MSELoss()                       # the target label is not one-hotted

            wf_col_train, wf_clean_train = DenoTD.make_training_data(n_train)
            train = Data.TensorDataset(torch.FloatTensor(wf_col_train), torch.FloatTensor(wf_clean_train))
            train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

            wf_col_test, wf_clean_test = DenoTD.make_training_data(n_test)

            # training and testing
            for epoch in range(EPOCH):
                for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                    est = self(b_x.cuda())[0] 
                    loss = loss_func(est, b_y.cuda())   # cross entropy loss b_y.cuda()
                    optimizer.zero_grad()           # clear gradients for this training step
                    loss.backward()                 # backpropagation, compute gradients
                    optimizer.step()                # apply gradients

                    if step % 100 == 0:
                        est_test = self(torch.FloatTensor(wf_col_test).cuda())[0]
                        l2_loss = np.mean(np.square(est_test.cpu().data.numpy() - wf_clean_test))
                        print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.4f' % l2_loss)

            # save model
            torch.save(self.state_dict(), fname_save)
        

    class Denoising_Training_Data(object):
        """
        templates obtained by simple clustering + averaging waveforms + crop and align
        spatial_sig, temporal_sig obtained from spikeinterface.sortingcomponents.waveform_tools.noise_whitener
        """

        def __init__(self,
                     templates, #Crop and denoise templates here
                     channel_index,
                     spatial_sig,
                     temporal_sig,
                     geom_array):

            self.spatial_sig = spatial_sig
            self.temporal_sig = temporal_sig
            self.templates = templates
            self.channel_index = channel_index
            self.spike_size = self.temporal_sig.shape[0]
            self.geom = geom_array
            
            self.templates, self.n_before, self.n_after = crop_and_align_templates(self.templates, self.spike_size, self.channel_index, self.geom)
            self.templates = denoise_templates(self.templates)
            
            self.templates = self.templates.transpose(0,2,1).reshape(
                -1, self.templates.shape[1])
            self.remove_small_templates()
            self.standardize_templates()
            self.jitter_templates()
            
            

        def remove_small_templates(self):

            ptp = self.templates.ptp(1)
            self.templates = self.templates[ptp > 3]

        def standardize_templates(self):

            # standardize templates
            ptp = self.templates.ptp(1)
            self.templates = self.templates/ptp[:, None]

            ref = np.mean(self.templates, 0)
            shifts = align_get_shifts_with_ref(
                self.templates, ref)
            self.templates = shift_chans(self.templates, shifts)

        def jitter_templates(self, up_factor=8):

            n_templates, n_times = self.templates.shape

            # upsample best fit template
            up_temp = scipy.signal.resample(
                x=self.templates,
                num=n_times*up_factor,
                axis=1)
            up_temp = up_temp.T

            idx = (np.arange(0, n_times)[:,None]*up_factor + np.arange(up_factor))
            up_shifted_temps = up_temp[idx].transpose(2,0,1)
            up_shifted_temps = np.concatenate(
                (up_shifted_temps,
                 np.roll(up_shifted_temps, shift=1, axis=1)),
                axis=2)
            self.templates = up_shifted_temps.transpose(0,2,1).reshape(-1, n_times)

            ref = np.mean(self.templates, 0)
            shifts = align_get_shifts_with_ref(
                self.templates, ref, upsample_factor=1)
            self.templates = shift_chans(self.templates, shifts)

        def make_training_data(self, n):

            n_templates, n_times = self.templates.shape

            center = n_times//2
            t_idx_in = slice(center - self.spike_size//2,
                             center + (self.spike_size//2) + 1)

            # sample templates
            idx1 = np.random.choice(n_templates, n)
            idx2 = np.random.choice(n_templates, n)
            wf1 = self.templates[idx1]
            wf2 = self.templates[idx2]

            # sample scale
            s1 = np.exp(np.random.randn(n)*0.8 + 2)
            s2 = np.exp(np.random.randn(n)*0.8 + 2)

            # turn off some
            c1 = np.random.binomial(1, 1-0.05, n)
            c2 = np.random.binomial(1, 1-0.05, n)

            # multiply them
            wf1 = wf1*s1[:, None]*c1[:, None]
            wf2 = wf2*s2[:, None]*c2[:, None]

            # choose shift amount
            shift = np.random.randint(low=0, high=3, size=(n,))

            # choose shift amount
            shift2 = np.random.randint(low=5, high=self.spike_size, size=(n,))

            shift *= np.random.choice([-1, 1], size=n)
            shift2 *= np.random.choice([-1, 1], size=n, p=[0.2, 0.8])

            # make colliding wf    
            wf_clean = np.zeros(wf1.shape)
            for j in range(n):
                temp = np.roll(wf1[j], shift[j])
                wf_clean[j] = temp

            # make colliding wf    
            wf_col = np.zeros(wf2.shape)
            for j in range(n):
                temp = np.roll(wf2[j], shift2[j])
                wf_col[j] = temp

            noise_wf = make_noise(n, self.spatial_sig, self.temporal_sig)[:, :, 0]

            wf_clean = wf_clean[:, t_idx_in]
            return (wf_clean + wf_col[:, t_idx_in] + noise_wf,
                    wf_clean)


        
        
        
        
def denoise_wf_nn_single_channel(wf, denoiser, device):
    """
    This function NN-denoises waveform arrays 
    TODO: avoid sending back and forth
    """
    assert(HAVE_TORCH)
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0] > 0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros_like(wf)

    return denoised_wf


def load_nn_and_denoise(wf_array, denoiser_weights_path, architecture_path):
    """
    This function Load NN and NN-denoises waveform arrays 
    TODO: Delete this function?
    """

    assert(HAVE_TORCH)
    architecture_denoiser = np.load(architecture_path, allow_pickle = True)

    model = SingleChanDenoiser(denoiser_weights_path,
                    architecture_denoiser['n_filters'], 
                    architecture_denoiser['filter_sizes'], 
                    architecture_denoiser['spike_size'])
    denoiser = model.load()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return denoise_wf_nn_single_channel(wf_array, denoiser, device)

def shift_chans(wf, best_shifts):
    ''' 
    Align all waveforms on a single channel given shifts
    '''
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    wfs_final= np.zeros(wf.shape, 'float32')
    for k, shift_ in enumerate(best_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final[k] = temp
    
    return wfs_final

def align_get_shifts_with_ref(wf, ref=None, upsample_factor=5, nshifts=7):

    ''' Returns shifts for aligning all waveforms on a single channel (ref)
    
        Used to generate training data
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    n_data, n_time = wf.shape

    if ref is None:
        ref = np.mean(wf, axis=0)
      
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1

    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = nshifts//2
    wf_end = -nshifts//2
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-(nshifts//2), (nshifts//2)+1)):
        ref_shifted[:,i] = ref_upsampled[s + wf_start: s + wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces

def make_noise(n, spatial_SIG, temporal_SIG):
    """Make noise
    Parameters
    ----------
    n: int
        Number of noise events to generate
    Returns
    ------
    numpy.ndarray
        Noise
    """
    n_neigh, _ = spatial_SIG.shape
    waveform_length, _ = temporal_SIG.shape

    # get noise
    noise = np.random.normal(size=(n, waveform_length, n_neigh))

    for c in range(n_neigh):
        noise[:, :, c] = np.matmul(noise[:, :, c], temporal_SIG)
        reshaped_noise = np.reshape(noise, (-1, n_neigh))

    the_noise = np.reshape(np.matmul(reshaped_noise, spatial_SIG),
                           (n, waveform_length, n_neigh))

    return the_noise


def order_channels_by_distance(reference, channels, geom):
    """Order channels by distance using certain channel as reference
    Parameters
    ----------
    reference: int
        Reference channel
    channels: np.ndarray
        Channels to order
    geom
        Geometry matrix
    Returns
    -------
    numpy.ndarray
        1D array with the channels ordered by distance using the reference
        channels
    numpy.ndarray
        1D array with the indexes for the ordered channels
    """
    coord_main = geom[reference]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))

    return channels[idx], idx


def crop_and_align_templates(templates, spike_size, channel_index, geom):
    """Crop (spatially) and align (temporally) templates
    Parameters
    ----------
    Returns
    -------
    """
    
    n_units, n_times, n_channels = templates.shape
    mcs = templates.min(1).argmin(1)
    
    ########## TEMPORALLY ALIGN TEMPLATES #################
    
    # template on max channel only
    templates_max_channel = np.zeros((n_units, n_times))
    for k in range(n_units):
        templates_max_channel[k] = templates[k, :, mcs[k]]

    # align them
    ref = np.mean(templates_max_channel, axis=0)
    upsample_factor = 8
    nshifts = spike_size//2

    shifts = align_get_shifts_with_ref(
        templates_max_channel, ref, upsample_factor, nshifts)

    templates_aligned = shift_chans(templates, shifts)
    
    # crop out the edges since they have bad artifacts
    templates_aligned = templates_aligned[:, nshifts//2:-nshifts//2]

    ########## Find High Energy Center of Templates #################

    templates_max_channel_aligned = np.zeros((n_units, templates_aligned.shape[1]))
    for k in range(n_units):
        templates_max_channel_aligned[k] = templates_aligned[k, :, mcs[k]]

    # determine temporal center of templates and crop around it
    total_energy = np.sum(np.square(templates_max_channel_aligned), axis=0)
    center = np.argmax(np.convolve(total_energy, np.ones(spike_size//2), 'same')) # Where maximum energy is 
        
    n_before_trough = spike_size//2 - (center - templates_max_channel_aligned.mean(0).argmin())
    n_after_trough = spike_size//2 + (center - templates_max_channel_aligned.mean(0).argmin())
        
    templates_aligned_bis = np.zeros((n_units, spike_size, templates_aligned.shape[2]))

    n_units_bad = 0
    for i in range(n_units):
        trough_i = templates_max_channel_aligned[i].argmin()
        if trough_i-n_before_trough>0 and trough_i+n_after_trough+1 < templates_aligned.shape[1]:
            templates_aligned_bis[i] = templates_aligned[i, trough_i-n_before_trough:trough_i+n_after_trough+1]
        else:
            n_units_bad+=1
    templates_aligned_bis = templates_aligned_bis[templates_aligned_bis.ptp(1).max(1)>0]
    ########## spatially crop (only keep neighbors) #################

    neighbors = channel_index #0/1 mask array shape 32*32#Channel Index 
    n_neigh = np.max(np.sum(neighbors, axis=1))
    templates_cropped = np.zeros((n_units-n_units_bad, spike_size, n_neigh))

    for k in range(n_units-n_units_bad):
        # get neighbors for the main channel in the kth template
        ch_idx = np.where(neighbors[mcs[k]])[0]

        # order channels
        ch_idx, _ = order_channels_by_distance(mcs[k], ch_idx, geom)
        
        # new kth template is the old kth template by keeping only
        # ordered neighboring channels
        templates_cropped[k, :, :ch_idx.shape[0]] = templates_aligned_bis[k][:, ch_idx]

    return templates_cropped, n_before_trough, n_after_trough

def denoise_templates(templates):

    n_templates, n_times, n_chan = templates.shape

    # remove templates with ptp < 5 (if there are enough templates)
    ptps = templates.ptp(1).max(1)
    if np.sum(ptps > 5) > 100:
        templates = templates[ptps>5]
        n_templates = templates.shape[0]

    denoised_templates = np.zeros(templates.shape)

    #templates on max channels (index 0)
    templates_mc = templates[:, :, 0]
    ptp_mc = templates_mc.ptp(1)
    templates_mc = templates_mc/ptp_mc[:, None]

    # denoise max channel templates
    # bug fix = PCA(n_components=5); sometimes test dataset may have too few templates... not realistic though
    pca_mc = PCA(n_components=min(min(templates_mc.shape[0], 
                                      templates_mc.shape[1]),
                                  5))
    score = pca_mc.fit_transform(templates_mc)
    deno_temp = pca_mc.inverse_transform(score)
    denoised_templates[:, :, 0] = deno_temp*ptp_mc[:, None]

    # templates on neighboring channels
    templates_neigh = templates[:, :, 1:]
    templates_neigh = templates_neigh.transpose(0, 2, 1).reshape(-1, n_times)
    ptp_neigh = templates_neigh.ptp(1)
    idx_non_zero = ptp_neigh > 0

    # get pca trained
    pca_neigh = PCA(n_components=5)
    pca_neigh.fit(templates_neigh[idx_non_zero]/ptp_neigh[idx_non_zero][:, None])

    # denoise them
    for j in range(1, n_chan):
        temp = templates[:, :, j]
        temp_ptp = np.abs(temp.min(1))
        idx = temp_ptp > 0
        if np.any(idx):
            temp = (temp[idx]/temp_ptp[idx, None])
            denoised_templates[idx, :, j] = pca_neigh.inverse_transform(
                pca_neigh.transform(temp))*temp_ptp[idx, None]

    return denoised_templates


def kill_signal(recordings, threshold, window_size):
    """
    Thresholds recordings, values above 'threshold' are considered signal
    (set to 0), a window of size 'window_size' is drawn around the signal
    points and those observations are also killed
    Returns
    -------
    recordings: numpy.ndarray
        The modified recordings with values above the threshold set to 0
    is_noise_idx: numpy.ndarray
        A boolean array with the same shap as 'recordings' indicating if the
        observation is noise (1) or was killed (0).
    """
    recordings = np.copy(recordings)

    T, C = recordings.shape
    R = int((window_size-1)/2)

    # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
    # recordings
    is_noise_idx = np.zeros((T, C))

    # go through every neighboring channel
    for c in range(C):

        # get obserations where observation is above threshold
        idx_temp = np.where(np.abs(recordings[:, c]) > threshold)[0]

        if len(idx_temp) == 0:
            is_noise_idx[:, c] = 1
            continue

        # shift every index found
        for j in range(-R, R+1):

            # shift
            idx_temp2 = idx_temp + j

            # remove indexes outside range [0, T]
            idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
                                                 idx_temp2 < T)]

            # set surviving indexes to nan
            recordings[idx_temp2, c] = np.nan

        # noise indexes are the ones that are not nan
        # FIXME: compare to np.nan instead
        is_noise_idx_temp = (recordings[:, c] == recordings[:, c])

        # standarize data, ignoring nans
        recordings[:, c] = recordings[:, c]/np.nanstd(recordings[:, c])

        # set non noise indexes to 0 in the recordings
        recordings[~is_noise_idx_temp, c] = 0

        # save noise indexes
        is_noise_idx[is_noise_idx_temp, c] = 1

    return recordings, is_noise_idx


def noise_whitener(recordings, temporal_size, window_size, sample_size=1000,
                   threshold=3.0, max_trials_per_sample=1000,
                   allow_smaller_sample_size=False):
    """Compute noise temporal and spatial covariance
    Parameters
    ----------
    recordings: numpy.ndarray
        Recordings
    temporal_size:
        Waveform size
    sample_size: int
        Number of noise snippets of temporal_size to search
    threshold: float
        Observations below this number are considered noise
    Returns
    -------
    spatial_SIG: numpy.ndarray
    temporal_SIG: numpy.ndarray
    """

    # kill signal above threshold in recordings
    rec, is_noise_idx = kill_signal(recordings, threshold, window_size)

    # compute spatial covariance, output: (n_channels, n_channels)
    spatial_cov = np.divide(np.matmul(rec.T, rec),
                            np.matmul(is_noise_idx.T, is_noise_idx))

    # compute spatial sig
    w_spatial, v_spatial = np.linalg.eig(spatial_cov)
    spatial_SIG = np.matmul(np.matmul(v_spatial,
                                      np.diag(np.sqrt(w_spatial))),
                            v_spatial.T)

    # apply spatial whitening to recordings
    spatial_whitener = np.matmul(np.matmul(v_spatial,
                                           np.diag(1/np.sqrt(w_spatial))),
                                 v_spatial.T)
    #print ("rec: ", rec, ", spatial_whitener: ", spatial_whitener.shape)
    rec = np.matmul(rec, spatial_whitener)

    # search single noise channel snippets
    noise_wf = search_noise_snippets(
        rec, is_noise_idx, sample_size,
        temporal_size,
        channel_choices=None,
        max_trials_per_sample=max_trials_per_sample,
        allow_smaller_sample_size=allow_smaller_sample_size)

    w, v = np.linalg.eig(np.cov(noise_wf.T))

    temporal_SIG = np.matmul(np.matmul(v, np.diag(np.sqrt(w))), v.T)

    return spatial_SIG, temporal_SIG


def search_noise_snippets(recordings, is_noise_idx, sample_size,
                          temporal_size, channel_choices=None,
                          max_trials_per_sample=1000,
                          allow_smaller_sample_size=False):
    """
    Randomly search noise snippets of 'temporal_size'
    Parameters
    ----------
    channel_choices: list
        List of sets of channels to select at random on each trial
    max_trials_per_sample: int, optional
        Maximum random trials per sample
    allow_smaller_sample_size: bool, optional
        If 'max_trials_per_sample' is reached and this is True, the noise
        snippets found up to that time are returned
    Raises
    ------
    ValueError
        if after 'max_trials_per_sample' trials, no noise snippet has been
        found this exception is raised
    Notes
    -----
    Channels selected at random using the random module from the standard
    library (not using np.random)
    """
    
    T, C = recordings.shape

    if channel_choices is None:
        noise_wf = np.zeros((sample_size, temporal_size))
    else:
        lenghts = set([len(ch) for ch in channel_choices])

        if len(lenghts) > 1:
            raise ValueError('All elements in channel_choices must have '
                             'the same length, got {}'.format(lenghts))

        n_channels = len(channel_choices[0])
        noise_wf = np.zeros((sample_size, temporal_size, n_channels))

    count = 0

    trial = 0

    # repeat until you get sample_size noise snippets
    while count < sample_size:

        # random number for the start of the noise snippet
        t_start = np.random.randint(T-temporal_size)

        if channel_choices is None:
            # random channel
            ch = random.randint(0, C - 1)
        else:
            ch = random.choice(channel_choices)

        t_slice = slice(t_start, t_start+temporal_size)

        # get a snippet from the recordings and the noise flags for the same
        # location
        snippet = recordings[t_slice, ch]
        snipped_idx_noise = is_noise_idx[t_slice, ch]

        # check if all observations in snippet are noise
        if snipped_idx_noise.all():
            # add the snippet and increase count
            noise_wf[count] = snippet
            count += 1
            trial = 0

        trial += 1

        if trial == max_trials_per_sample:
            if allow_smaller_sample_size:
                return noise_wf[:count]
            else:
                raise ValueError("Couldn't find snippet {} of size {} after "
                                 "{} iterations (only {} found)"
                                 .format(count + 1, temporal_size,
                                         max_trials_per_sample,
                                         count))

    return noise_wf
