from pydub import AudioSegment
import os

# set the duration of the sample in milliseconds
sample_duration = 4000

# set the input and output directories
input_dir = "highway"
output_dir = "highway4second"

# loop through all the files in the input directory
for filename in os.listdir(input_dir):
    # check if the file is a WAV file
    if filename.endswith(".wav"):
        # load the audio file using pydub
        audio = AudioSegment.from_wav(os.path.join(input_dir, filename))
        
        # calculate the number of 4-second-long samples that can be extracted
        num_samples = len(audio) // sample_duration
        
        # loop through each sample and export it as a new file
        for i in range(num_samples):
            # calculate the start and end times of the sample
            start_time = i * sample_duration
            end_time = start_time + sample_duration
            
            # extract the sample
            sample = audio[start_time:end_time]
            
            # create a new filename for the sample
            sample_filename = os.path.splitext(filename)[0] + f"_sample{i+1}.wav"
            
            # export the sample to the output directory
            sample.export(os.path.join(output_dir, sample_filename), format="wav")
