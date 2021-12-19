import os
import librosa
import soundfile as sf

def create_one_second_noise_files():
    """
    Supporting function that takes the recorded noise and extract 
    one-second-samples.
    Stores samples in folder as wav.
    """
    
    for file in os.listdir(NOISE_FOLDER):
        if not file.endswith('.wav'):
            continue
        wav, sr = librosa.load(NOISE_FOLDER + "/" + file)
        wav_length = int(len(wav) / sr)
        
        for second_start in range(wav_length):
            second_end = (second_start+1)*sr
            one_second_noise = wav[second_start*sr:second_end]
            sf.write(CLIP_FOLDER+'/'+file.split('.')[0]+'_'+str(second_start)+'.wav', 
                     one_second_noise, 
                     22050, 
                     subtype='PCM_16'
                    )
    
if __name__ == "__main__":
    # Defining global variables
    NOISE_FOLDER = 'background_noise'
    CLIP_FOLDER = NOISE_FOLDER+'/one_second_clips'

    create_one_second_noise_files()