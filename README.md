Deep learning classifier for Sleep Apnea diagnosis of video files of patients sleeping, that can then be linked to underlying conditions.


Prerequisites:

Either use patient separated video files of your own or use the supplied test data from an open-source sleep study:

https://figshare.com/articles/dataset/sleep_dataset_zip/5518996




Includes:

- Frame preprocessor to play through video files, choose frames of interest (in this case unique facial  frames with significant motion) and save each patient's frames in split subdirectories .
- Annotation tool to manually annotate the processed frames, by shuffling through the directories with frames and saving annotations in .txt file.
- Finally, the frames and annotations can be processed and loaded into the neural network and used as test and training data to build out a classifier to diagnose sleep apnea, as well as performance graphs like PR curves and accuracy.
