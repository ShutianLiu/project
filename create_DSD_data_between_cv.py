
# coding: utf-8

# In[1]:

import numpy as np
import librosa
import os
import sys
import h5py


# In[6]:

# create a wav file dataset

def create_DSD_dataset(mode='tr', sr=44100, nfft=2048, nhop=512, 
                       use_mel=True, mel_filt=250, target='wiener'):
    
    file_list = []
    
    cal_stat = False
    
    load_stat = True
    
    #define_path = '/hdd1/yluo/data/DSD100_Chimera/'+str(sr)+'/'+str(nfft)+'/'+str(nhop)
    define_path = '/mnt/f/research/data/SiSEC/DSD100/BetweenClassFeature/'+str(sr)+'/'+str(nfft)+'/'+str(nhop)
    
    if not os.path.exists(define_path):
        os.makedirs(define_path)
    
    data_path = define_path+'/'+'stat_'+str(mel_filt)
    
    if mode is 'tr':
        #mixing_list = '/hdd1/yluo/data/DSD100_Chimera/mix_list_train.txt'
        mixing_list = '/mnt/f/research/data/SiSEC/DSD100/dev_list.txt' 
        with open(mixing_list) as f:
            for lines in f:
                file_list.append(lines.rstrip('\n'))   #file_list contains names of files to mix
        
        new_set = h5py.File(define_path+'/data_tr_'+str(mel_filt))
        cal_stat = True
        load_stat = False
        #########
        folderpath='/Sources/Dev/'
        
    elif mode is 'cv':
        #mixing_list = '/hdd1/yluo/data/DSD100_Chimera/mix_list_cv.txt'
        mixing_list = '/mnt/f/research/data/SiSEC/DSD100/test_list.txt' 
        with open(mixing_list) as f:
            for lines in f:
                file_list.append(lines.rstrip('\n'))
                
        new_set = h5py.File(define_path+'/data_cv_'+str(mel_filt))
        ###########
        folderpath='/Sources/Test/'
    else:
        raise NotImplementedError
        
    num_data = len(file_list)   #total number of files to mix
    print('TOtAL file number :')
    print(num_data)
    
    checkpoint = num_data // 20
    
    nframe = 0
    #####
    num_saved = 0   #total number of data saved in file 

    if load_stat: #when making training data, its false
        # load 
        mean, var = np.load(data_path+'.npy')
        print('mean, var  loaded from .npy :')
        print(mean, var)


    for i in xrange(num_data):

        print('FIle number =')
        print(i)
        print('File Name =')
        print(file_list[i])

        if (i+1) % checkpoint == 0:
            print str(5*(i+1)/checkpoint)+'% finished...'
            
        #cur_name = file_list[i].split(".wav ")  #pick the name before'.wav'
        #vocal = cur_name[0]+'.wav'               
        #bgm = cur_name[1]
        cur_name = file_list[i]

        
        # read file using single precision
        #vocal, e_sr = librosa.load(vocal, sr=44100)   #load vocal and bgm wave file
        #bgm, _ = librosa.load(bgm, sr=44100)
        wavepath='/mnt/f/research/data/SiSEC/DSD100'+folderpath+str(file_list[i])
        wavepath=wavepath.split("\r")[0]
        vocals, e_sr = librosa.load(str(wavepath+'/vocals.wav'), sr=44100)
        bass , _ =  librosa.load(str(wavepath+'/bass.wav'), sr=44100)
        drums , _ =  librosa.load(str(wavepath+'/drums.wav'), sr=44100)
        other , _ = librosa.load(str(wavepath+'/other.wav'), sr=44100)

        if e_sr != sr:
            # downsample           
            #vocal = librosa.resample(vocal, e_sr, sr)
            #bgm = librosa.resample(bgm, e_sr, sr)
            vocals = librosa.resample(vocals, e_sr, sr)
            bass = librosa.resample(bass, e_sr, sr)
            drums = librosa.resample(drums, e_sr, sr)
            other = librosa.resample(other, e_sr, sr)
        # rescaling
        #vocal = vocal * 0.9 / np.max(vocal)
        #bgm = bgm * 0.9 / np.max(bgm)
        vocals = vocals * 0.9 / np.max(vocals)
        bass = bass * 0.9 / np.max(bass)
        drums = drums * 0.9 / np.max(drums)
        other = other * 0.9 / np.max(other)

        length = vocals.shape[0]
        
        vocals = vocals[:length]      #what is done here???
        bass = bass[:length]
        drums = drums[:length]
        other = other[:length]
        
        mix = vocals + bass + drums + other
        
        # generate features
        # need: input feature, phase, spectrograms, hard mask
        
        mix_comp = librosa.stft(mix, n_fft=nfft, hop_length=nhop, dtype=np.complex64).T  # (T, F) #complex
        #phase_mix = np.angle(mix_comp)
        mix_spec = np.abs(mix_comp) 
        
        v_comp = librosa.stft(vocals, n_fft=nfft, hop_length=nhop, dtype=np.complex64).T
        #phase_v = np.angle(v_comp)
        v_spec = np.abs(v_comp)
        
        b_comp = librosa.stft(bass, n_fft=nfft, hop_length=nhop, dtype=np.complex64).T
        #phase_b = np.angle(b_comp)
        b_spec = np.abs(b_comp)

        d_comp = librosa.stft(drums, n_fft=nfft, hop_length=nhop, dtype=np.complex64).T
        #phase_d = np.angle(d_comp)
        d_spec = np.abs(d_comp)

        o_comp = librosa.stft(other, n_fft=nfft, hop_length=nhop, dtype=np.complex64).T
        #phase_o = np.angle(o_comp)
        o_spec = np.abs(o_comp)
        
        # input feature
        
        if use_mel:
            # mel filterbank
            mel = librosa.filters.mel(sr, nfft, mel_filt)
        
            infeat = np.dot(mel, mix_spec.T).T
            mix_spec = np.dot(mel, mix_spec.T).T
            v_spec = np.dot(mel, v_spec.T).T
            b_spec = np.dot(mel, b_spec.T).T
            d_spec = np.dot(mel, d_spec.T).T
            o_spec = np.dot(mel, o_spec.T).T
        else:
            # log spec in dB
            infeat = 20 * np.log10(mix_spec)
            infeat = np.maximum(infeat, -50)
        
        T, F = infeat.shape
        
        if i == 0:
            print T, F
        
        nframe += T
        
        # hard mask
        
        all_s = np.zeros((4, T*F), dtype=np.float32)

        all_s[0,:] = v_spec.reshape((-1,))
        all_s[1,:] = b_spec.reshape((-1,))
        all_s[2,:] = d_spec.reshape((-1,))
        all_s[3,:] = o_spec.reshape((-1,))

        idx = np.argmax(all_s, axis=0).reshape(T, F)  #this is the Y matrix in the paper
        hard_mask = np.zeros((4, T, F))

        for j in range(4):            #hard mask for 2 sources
            tp = np.copy(idx)
            tp[tp!=j] = -1
            tp[tp==j] = 1
            tp[tp==-1] = 0
            hard_mask[j,:,:] = tp
            
        hard_mask = hard_mask.transpose(1,2,0)  # (T, F, 2)
        
        # target mask
        
        if target is 'soft':
            soft_mask = np.zeros((4, T, F))
            soft_mask[0] = v_spec / (v_spec + b_spec + d_spec + o_spec)
            soft_mask[1] = b_spec / (v_spec + b_spec + d_spec + o_spec)
            soft_mask[2] = d_spec / (v_spec + b_spec + d_spec + o_spec)
            soft_mask[3] = o_spec / (v_spec + b_spec + d_spec + o_spec)
            soft_mask[np.isnan(soft_mask)] = 0.
        
        elif target is 'wiener':
            soft_mask = np.zeros((4, T, F))
            soft_mask[0] = v_spec**2 / (v_spec**2 + b_spec**2 + d_spec**2 + o_spec**2)
            soft_mask[1] = b_spec**2 / (v_spec**2 + b_spec**2 + d_spec**2 + o_spec**2)
            soft_mask[2] = d_spec**2 / (v_spec**2 + b_spec**2 + d_spec**2 + o_spec**2)
            soft_mask[3] = o_spec**2 / (v_spec**2 + b_spec**2 + d_spec**2 + o_spec**2)
            soft_mask[np.isnan(soft_mask)] = 0.
        
        soft_mask = soft_mask.transpose(1,2,0)
        
        # save to file

        n_100frame =  T // 100

        if i == 0: 
            maxshape = (100000, 100, F)
            shape = (num_data, T, F)
            if mode is 'tr':
                shape = (10000, 100, F )
            if mode is 'cv':
                shape = (10000, 100, F )

            infeat_set = new_set.create_dataset('infeat', shape=shape, maxshape=maxshape,
                                             dtype=np.float32)
        
            shape = (num_data, T, F)
            if mode is 'tr':
                shape = (10000, 100, F )
            if mode is 'cv':
                shape = (10000, 100, F )
        
            mix_spec_set = new_set.create_dataset('mix_spec', shape=shape, maxshape=maxshape,
                                         dtype=np.float32)
            
            maxshape = (100000, 100, F, 4)
            shape = (num_data, T, F, 4)
            if mode is 'tr':
                shape = (10000, 100, F, 4 )
            if mode is 'cv':
                shape = (10000, 100, F, 4)

            hard_mask_set = new_set.create_dataset('hard_mask', shape=shape, maxshape=maxshape,
                                         dtype=np.float32)
            
            target_mask_set = new_set.create_dataset('target_mask', shape=shape, maxshape=maxshape,
                                         dtype=np.float32)
            
            if cal_stat:
                mean = np.zeros(F, dtype=np.float32)
                var = np.zeros(F, dtype=np.float32)
                mean += np.sum(infeat[:T,:], axis=0)
                
            if load_stat:
                infeat[:T,:] = (infeat[:T,:] - mean) / np.sqrt(var+1e-8)
        
        else:
            if cal_stat:
                mean += np.sum(infeat[:T,:], axis=0)
            
            if load_stat:
                infeat[:T,:] = (infeat[:T,:] - mean) / np.sqrt(var+1e-8)
        print('shape of infeat:')  
        print(infeat.shape)  
        print('T and F:')
        print(T,F)
        ################
        for k in range(n_100frame):    
            infeat_set[num_saved,k*100:(k+1)*100,:] = infeat[k*100:(k+1)*100,:]
            mix_spec_set[num_saved,k*100:(k+1)*100,:] = mix_spec[k*100:(k+1)*100,:]
            hard_mask_set[num_saved,:100] = hard_mask[k*100:(k+1)*100]
            target_mask_set[num_saved,:100] = soft_mask[k*100:(k+1)*100]

            num_saved += 1
        
    if cal_stat:
        mean /= nframe
        #for i in range(num_data):
        for i in range(num_saved):
            current_infeat = infeat_set[i]
            #current_infeat = current_infeat[:T,:]
            current_infeat = current_infeat[:100,:]
            var += np.sum((current_infeat - mean)**2, axis=0)
        var /= nframe
        
        # apply it to the dataset
        #for i in range(num_data):
            #infeat_set[i,:T,:] = (infeat_set[i,:T,:] - mean) / np.sqrt(var+1e-8)
        for i in range(num_saved):
            infeat_set[i,:100,:]= (infeat_set[i,:100,:] - mean) / np.sqrt(var+1e-8)
            
        np.save(data_path, [mean, var])
        infeat_set.attrs['stats'] = [mean, var]
        infeat_set.attrs['sr'] = sr
        infeat_set.attrs['mel'] = mel_filt
        infeat_set.attrs['stft'] = [nfft, nhop]
            
    new_set.close()
    print 'Finished.'


# In[7]:

#create_DSD_dataset(mode='tr', sr=16000, nfft=1024, nhop=256, mel_filt=150)
create_DSD_dataset(mode='cv', sr=16000, nfft=1024, nhop=256, mel_filt=150)

#create_DSD_dataset(mode='tr', sr=16000, nfft=1024, nhop=256, mel_filt=200)
#create_DSD_dataset(mode='cv', sr=16000, nfft=1024, nhop=256, mel_filt=200)

#create_DSD_dataset(mode='tr', sr=22050, nfft=1024, nhop=256, mel_filt=200)
#create_DSD_dataset(mode='cv', sr=22050, nfft=1024, nhop=256, mel_filt=200)

#create_DSD_dataset(mode='tr', sr=22050, nfft=1024, nhop=256, mel_filt=250)
#create_DSD_dataset(mode='cv', sr=22050, nfft=1024, nhop=256, mel_filt=250)

#create_DSD_dataset(mode='tr', sr=22050, nfft=2048, nhop=512, mel_filt=250)
#create_DSD_dataset(mode='cv', sr=22050, nfft=2048, nhop=512, mel_filt=250)

#create_DSD_dataset(mode='tr', sr=22050, nfft=2048, nhop=512, mel_filt=300)
#create_DSD_dataset(mode='cv', sr=22050, nfft=2048, nhop=512, mel_filt=300)

