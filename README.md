* The codes_50, and rare_50 files contain the label list for the MIMIC-III-50, and the MIMIC-III-rare50 datasets respectively.
* Using the trian_concepts.py file, one can train the concept prediction LLM model for both rare/top-50 codes. User needs to specify the data subset, LLM, concept-level (parent or leaf) from lines 19 to 27.
* Running the test_concepts.py file should be done in a similar way. The dataset is available at (https://physionet.org/content/mimiciii/1.4/) by applying following the suggested procedure. We have to extract the splits for top-50 from (https://github.com/jamesmullenbach/caml-mimic), and for rare-50 from (https://github.com/whaleloops/KEPT)
* Change the data file type if necessary.
* The 'new_concepts_seq_to_label_fine_tuning.py' file does the fine-tuning task for the LLM to predict ICD labels from Concept Predictions.
* The 'run_refiner_inference_concepts_seq_to_label.py' uses the FT-ed model from the previous step to do the final labeling task.
